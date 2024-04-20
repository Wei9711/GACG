import copy
from .q_learner import QLearner
from components.episode_buffer import EpisodeBatch
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer
import torch
import torch as th
import torch.nn as nn
from torch.optim import RMSprop
import sys
import torch.nn.functional  as F
# from torch_geometric.nn import DiffGroupNorm
from torch.distributions import Categorical

class VASTNet(nn.Module):
    def __init__(self, nr_input_features, nr_subteams, nr_hidden_units=128):
        super(VASTNet, self).__init__()
        self.nr_input_features = nr_input_features
        self.nr_hidden_units = nr_hidden_units

        self.fc_net = nn.Sequential(
            nn.Linear(self.nr_input_features, self.nr_hidden_units),
            nn.ELU(),
            nn.Linear(self.nr_hidden_units, self.nr_hidden_units),
            nn.ELU()
        )

        # Adjust the output size of the action_head layer
        self.action_head = nn.Linear(self.nr_hidden_units, nr_subteams)

    def forward(self, x):
        # Change the dimensions to [batch_size * seq_len, input_features]
        batch_size, seq_len, input_features = x.size()
        x = x.view(-1, input_features)
        # Apply the fully connected layers
        x = self.fc_net(x)
        # Reshape back to [batch_size, seq_len, nr_hidden_units]
        x = x.view(batch_size, seq_len, self.nr_hidden_units)
        # # Sum over the sequence dimension
        x = self.action_head(x)
        # Apply the action_head layer
        output = F.softmax(x, dim=-1)
        return output


class VastQLearner(QLearner):
    def __init__(self, mac, scheme, logger, args):
        super(VastQLearner, self).__init__(mac, scheme, logger, args)
        self.args = args

        self.vast_input = args.state_shape + args.obs_shape
        self.vast_out = args.group_num
        self.VAST = VASTNet(self.vast_input,self.vast_out).cuda()


    def train(self, episode_sample: EpisodeBatch, max_ep_t, t_env: int, episode_num: int):
        # Get the relevant quantities
        batch = episode_sample[:, :max_ep_t]
        rewards = batch["reward"][:, :-1]
        # print("rewards", rewards.shape)

        actions = batch["actions"][:, :-1]
        # print("actions", actions.shape)

        terminated = batch["terminated"][:, :-1].float()
        # print("terminated", terminated.shape)
        # print("terminated", terminated[0])

        mask = batch["filled"][:, :-1].float()
        # print("mask", mask.shape)
        # print("mask", mask[0])
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        # print("mask", mask[0])


        avail_actions = batch["avail_actions"]
        # print("avail_actions", avail_actions.shape)
        # print("avail_actions", avail_actions[0,0,0,:])
        # print("avail_actions", avail_actions.sum())
        # sys.exit()
        # Calculate estimated Q-Values
        mac_out = []
        group_index_list = []
        hidden_states_list = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        
        mac_out = th.stack(mac_out, dim=1)  # Concat over time [32, 74, 8, 14]

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim

        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        # target_group_index_list = []
        # target_obs_mlp_out = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs = self.target_mac.forward(batch, t=t)
            target_mac_out.append(target_agent_outs)
            # target_group_index_list.append(target_group_index)
            # target_obs_mlp_out.append(obs_mlp_emb)

        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = th.stack(target_mac_out[1:], dim=1)  # Concat across time
        # print("target_mac_out",target_mac_out)
        # Mask out unavailable actions
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999
        # print("target_mac_out",target_mac_out)
        # Max over target Q-Values
        if self.args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]
        # print("target_mac_out",target_mac_out.shape)
        # print("target_max_qvals",target_max_qvals.shape)
        # Mix
        # print(batch["state"][:, :-1].shape)
        # print("chosen_action_qvals",chosen_action_qvals.shape)
        # print("target_max_qvals",target_max_qvals.shape)
        # print(cnn_state.shape)

        # -----------------Vast Sum Subgroup ---------------------------
        obs = batch["obs"] # [32, 86, 8, 128]
        # print("obs",obs.shape)
        vast_state = batch["state"].unsqueeze(2)
        vast_state = vast_state.repeat(1, 1, obs.size(2), 1)
        # print("vast_state",vast_state.shape)
        vast_input = torch.cat([vast_state.reshape(-1,self.args.n_agents,vast_state.size(-1)), obs.reshape(-1,self.args.n_agents,obs.size(-1))], dim=-1)
        # print("vast_input",vast_input.shape)
        sub_group_index = self.VAST(vast_input)
        # print("sub_group_index",sub_group_index.shape)
        assignment_dist = Categorical(sub_group_index)
        subteam_ids = assignment_dist.sample().detach()
        subteam_ids = subteam_ids.reshape(batch.batch_size,-1, self.args.n_agents)
        # print("subteam_ids",subteam_ids.shape)

        subg_chosen_action_qvals = torch.zeros(subteam_ids.size(0), subteam_ids.size(1)-1, 2).cuda()
        subg_target_max_qvals = torch.zeros(subteam_ids.size(0), subteam_ids.size(1)-1, 2).cuda()

        for subgroup_index in range(2):
            subgroup_mask = (subteam_ids == subgroup_index)
            subg_chosen_action_qvals[:, :, subgroup_index] = (chosen_action_qvals * subgroup_mask[:, :-1]).sum(dim=-1)
            subg_target_max_qvals[:, :, subgroup_index] = (target_max_qvals * subgroup_mask[:, 1:]).sum(dim=-1)
        # -----------------Vast Sum Subgroup ---------------------------

        # print("chosen_action_qvals",subg_chosen_action_qvals.shape)
        # print("target_max_qvals",subg_target_max_qvals.shape)
        # sys.exit()
        if self.mixer is not None:
            chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])
            target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:])
        # if self.mixer is not None:
        #     chosen_action_qvals = self.mixer(chosen_action_qvals, cnn_state[:, :-1])
        #     target_max_qvals = self.target_mixer(target_max_qvals, cnn_state[:, 1:])

        # Calculate 1-step Q-Learning targets
        # print("rewards",rewards.shape)
        # print("rewards",rewards[0])
        # print("terminated",terminated.shape)
        # print("terminated",terminated[0])
        
        targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals

        # Td-error
        td_error = (chosen_action_qvals - targets.detach())
        # print("td_error",td_error.shape)
        # print("td_error",td_error[0])
        mask = mask.expand_as(td_error)
        # print("mask",mask.shape)
        # print("mask",mask[0])
        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask


        loss = (masked_td_error ** 2).sum() / mask.sum()
        # sys.exit()
        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            # self.logger.log_stat("reward", th.sum(rewards).item(), t_env)
            self.logger.log_stat("loss", loss.item(), t_env)
            # if self.args.is_train_groupnizer:
            #     self.logger.log_stat('Gdistance_mean', Gdistance_mean, t_env)
            #     self.logger.log_stat("target_Gdistance", target_Gdistance, t_env)
            self.logger.log_stat("grad_norm", grad_norm.item(), t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
            self.logger.log_stat("q_taken_mean", (chosen_action_qvals * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            # if is_adv == False:
            #     self.logger.log_matrix("trj_emb", trj_emb, t_env)

            # if self.args.is_masssge and Atten_graph is not None:
            #     self.logger.log_matrix("Atten_adj", Atten_graph[0], t_env)
            #     if group_index is not None:
            #         self.logger.log_matrix('group_index', group_index_out[0], t_env)

            self.log_stats_t = t_env


    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        # self.trj_encoder.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))

    @staticmethod
    def _mlp(input, hidden_dims, output):
        """ Creates an MLP with the specified input and output dimensions and (optional) hidden layers. """
        hidden_dims = [] if hidden_dims is None else hidden_dims
        hidden_dims = [hidden_dims] if isinstance(hidden_dims, int) else hidden_dims
        dim = input
        layers = []
        for d in hidden_dims:
            layers.append(nn.Linear(dim, d))
            layers.append(nn.ReLU())
            dim = d

        layers.append(nn.Linear(dim, output))
        return (nn.Sequential)(*layers)
    
    @staticmethod
    def group_distance_ratio(x, y, eps= 1e-5):
        r"""Measures the ratio of inter-group distance over intra-group
        distance.

        .. math::
            R_{\text{Group}} = \frac{\frac{1}{(C-1)^2} \sum_{i!=j}
            \frac{1}{|\mathbf{X}_i||\mathbf{X}_j|} \sum_{\mathbf{x}_{iv}
            \in \mathbf{X}_i } \sum_{\mathbf{x}_{jv^{\prime}} \in \mathbf{X}_j}
            {\| \mathbf{x}_{iv} - \mathbf{x}_{jv^{\prime}} \|}_2 }{
            \frac{1}{C} \sum_{i} \frac{1}{{|\mathbf{X}_i|}^2}
            \sum_{\mathbf{x}_{iv}, \mathbf{x}_{iv^{\prime}} \in \mathbf{X}_i }
            {\| \mathbf{x}_{iv} - \mathbf{x}_{iv^{\prime}} \|}_2 }

        where :math:`\mathbf{X}_i` denotes the set of all nodes that belong to
        class :math:`i`, and :math:`C` denotes the total number of classes in
        :obj:`y`.
        """
        num_classes = int(y.max()) + 1

        numerator = 0.
        for i in range(num_classes):
            mask = y == i
            dist = torch.cdist(x[mask].unsqueeze(0), x[~mask].unsqueeze(0))
            numerator += (1 / (dist.numel()+ eps)) * float(dist.sum())
        numerator *= 1 / (num_classes - 1)**2

        denominator = 0.
        for i in range(num_classes):
            mask = y == i
            dist = torch.cdist(x[mask].unsqueeze(0), x[mask].unsqueeze(0))
            denominator += (1 / (dist.numel()+ eps)) * float(dist.sum())
        denominator *= 1 / num_classes

        return numerator / (denominator + eps)
