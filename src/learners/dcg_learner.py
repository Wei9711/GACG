from .q_learner import QLearner
from components.episode_buffer import EpisodeBatch
import torch as th
from torch.optim import RMSprop


class DCGLearner(QLearner):
    """ QLearner for a Deep Coordination Graph (DCG, Boehmer et al., 2020). """

    def __init__(self, mac, scheme, logger, args):
        super(DCGLearner, self).__init__(mac, scheme, logger, args)
        
        # action encoder
        self.use_action_repr = args.use_action_repr
        if self.use_action_repr:
            self.action_encoder_params = list(self.mac.action_encoder_params())
            self.action_encoder_optimiser = RMSprop(params=self.action_encoder_params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)
            self.action_repr_updating = True

    def train(self, episode_sample: EpisodeBatch, max_ep_t, t_env: int, episode_num: int):
        """ Overrides the train method from QLearner. """

        # Get the relevant quantities
        batch = episode_sample[:, :max_ep_t]
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])

        # Calculate the maximal Q-Values of the target network
        target_out = []
        self.target_mac.init_hidden(batch.batch_size)
        self.mac.init_hidden(batch.batch_size)
        # Run through the episodes in the batch step by step
        for t in range(batch.max_seq_length):
            # In double Q-learning, the actions are selected greedy w.r.t. mac
            greedy = self.mac.forward(batch, t=t, policy_mode=False)
            # Q-value of target_mac with the above greedy actions
            target_out.append(self.target_mac.forward(batch, t=t, actions=greedy, policy_mode=False))
        # The TD-targets for time steps 1 to max_seq_length-1 (i.e., one step in the future)
        target_out = th.stack(target_out[1:], dim=1).unsqueeze(dim=-1)  # Concat across time, starting at index 1
        targets = rewards + self.args.gamma * (1 - terminated) * target_out

        # Calculate estimated Q-Values for the current actions
        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        # Q-values from time step 0 to max_seq_length-2 (i.e., the present)
        for t in range(batch.max_seq_length - 1):
            val = self.mac.forward(batch, t=t, actions=actions[:, t], policy_mode=False, compute_grads=True)
            mac_out.append(val)
        mac_out = th.stack(mac_out, dim=1).unsqueeze(dim=-1)  # Concat the Q-values over time

        # Calculate TD-error and masked loss for 1-step Q-Learning targets
        td_error = (mac_out - targets.detach())
        mask = mask.expand_as(td_error)
        td_error = td_error * mask
        loss = (td_error ** 2).sum() / mask.sum()

        # Optimise the loss
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        
        # Learning action representation
        if self.use_action_repr:
            pred_obs_loss = None
            pred_r_loss = None
            pred_grad_norm = None
            if self.action_repr_updating:
            # train action encoder
                # predictions of next observation
                no_pred = []  
                # prediction of reward for each timestep t
                r_pred = []   
                for t in range(batch.max_seq_length):
                    no_preds, r_preds = self.mac.action_repr_forward(batch, t=t)
                    no_pred.append(no_preds)
                    r_pred.append(r_preds)
                no_pred = th.stack(no_pred, dim=1)[:, :-1]  # Concat over time
                r_pred = th.stack(r_pred, dim=1)[:, :-1]
                #  True observations
                no = batch["obs"][:, 1:].detach().clone()
                # True rewards
                repeated_rewards = batch["reward"][:, :-1].detach().clone().unsqueeze(2).repeat(1, 1, self.args.n_agents, 1)

                #Root mean square error (RMSE) for the observation prediction 
                #Mean square error (MSE) for the reward prediction
                pred_obs_loss = th.sqrt(((no_pred - no) ** 2).sum(dim=-1)).mean()
                pred_r_loss = ((r_pred - repeated_rewards) ** 2).mean()

                pred_loss = pred_obs_loss + 10 * pred_r_loss
                self.action_encoder_optimiser.zero_grad()
                pred_loss.backward()
                pred_grad_norm = th.nn.utils.clip_grad_norm_(self.action_encoder_params, self.args.grad_norm_clip)
                self.action_encoder_optimiser.step()

                if t_env > self.args.action_repr_learning_phase:
                    self.mac.update_action_repr()
                    self.action_repr_updating = False
                    self._update_targets()
                    self.last_target_update_episode = episode_num

        # Update target network if it is time
        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        # Log important learning variables
        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm.item(), t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (td_error.abs().sum().item()/mask_elems), t_env)
            self.logger.log_stat("q_taken_mean", (mac_out * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.log_stats_t = t_env

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")
        if self.use_action_repr:
            self.target_mac.action_repr_updating = self.action_repr_updating
