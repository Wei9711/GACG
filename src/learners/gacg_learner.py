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
# from torch_geometric.nn import DiffGroupNorm

class GroupQLearner(QLearner):
    def __init__(self, mac, scheme, logger, args):
        super(GroupQLearner, self).__init__(mac, scheme, logger, args)
        self.args = args
        self.group_start = args.obs_group_trunk_size
        self.group_loss_weight = args.group_loss_weight


    def train(self, episode_sample: EpisodeBatch, max_ep_t, t_env: int, episode_num: int):
        # Get the relevant quantities
        batch = episode_sample[:, :max_ep_t]
        rewards = batch["reward"][:, :-1]
        # print("rewards", rewards.shape)

        actions = batch["actions"][:, :-1]
        # print("actions", actions.shape)

        reshaped_tensor = episode_sample["obs"][:, :-1].clone().view(batch.batch_size, self.args.n_agents, -1)
        # print(reshaped_tensor.shape)
        # trj_emb = self.trj_encoder.forward(reshaped_tensor)

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
            agent_outs, Atten_graph, group_index, _ = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
            group_index_list.append(group_index)
            hidden_states_list.append(self.mac.hidden_states.reshape(batch.batch_size, self.args.n_agents, -1))
        

        # print(group_index.shape)
        # print(gcn_message.shape)
        # print(Gdistance)
        # sys.exit()
        mac_out = th.stack(mac_out, dim=1)  # Concat over time [32, 74, 8, 14]
        hidden_states_out = th.stack(hidden_states_list, dim=1) # [32, 74, 8, 64]
        group_index_out = th.stack((group_index_list[self.group_start:]), dim=1) #[32, 64, 8]
        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim

        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        # target_group_index_list = []
        # target_obs_mlp_out = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs,_,_, _ = self.target_mac.forward(batch, t=t)
            target_mac_out.append(target_agent_outs)
            # target_group_index_list.append(target_group_index)
            # target_obs_mlp_out.append(obs_mlp_emb)

        # if self.args.is_train_groupnizer:
        #     Gdistance = []
            # for t_g in range(1, group_index_out.shape[1]):
            #     for b in range(0,group_index_out.shape[0],8): #range(1, group_index_out.shape[1])
            #     # print(group_index_out[:,:, t_g-1].shape)
            #         if torch.any(group_index_out[b, t_g-1] != group_index_out[b, t_g]): # Group index change
            #             num_group = group_index_out[b, t_g].max() + 1
            #             if num_group > 1:
            #                 G_temp = self.group_distance_ratio(mac_out[b, t_g + self.group_start], group_index_out[b, t_g])
            #                 Gdistance.append(G_temp)
            #             else:
            #                 Gdistance.append(1)
        if self.args.is_train_groupnizer:
            Gdistance = []
            for t_g in range(group_index_out.shape[1]):
                num_group = group_index_out[:, t_g].max() + 1
                if num_group > 1:
                    G_temp = self.group_distance_ratio(mac_out[:, t_g + self.group_start], group_index_out[:, t_g])
                    Gdistance.append(G_temp)
                else:
                    Gdistance.append(1)

            # if len(Gdistance) > 0:
            Gdistance_mean = sum(Gdistance) / len(Gdistance)



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

        # print(cnn_state.shape)
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

        if self.args.is_train_groupnizer:
                loss = (masked_td_error ** 2).sum() / mask.sum() - self.group_loss_weight * Gdistance_mean
        else:
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
            if self.args.is_train_groupnizer:
                self.logger.log_stat('Gdistance_mean', Gdistance_mean, t_env)
            #     self.logger.log_stat("target_Gdistance", target_Gdistance, t_env)
            self.logger.log_stat("grad_norm", grad_norm.item(), t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
            self.logger.log_stat("q_taken_mean", (chosen_action_qvals * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            # if is_adv == False:
            #     self.logger.log_matrix("trj_emb", trj_emb, t_env)

            if self.args.is_masssge and Atten_graph is not None:
                self.logger.log_matrix("Atten_adj", Atten_graph[0], t_env)
                if group_index is not None:
                    self.logger.log_matrix('group_index', group_index_out[0], t_env)

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
