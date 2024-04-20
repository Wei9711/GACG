# uncompyle6 version 3.9.0
# Python bytecode version base 3.7.0 (3394)
# Decompiled from: Python 3.9.12 (main, Apr  5 2022, 06:56:58) 
# [GCC 7.5.0]
# Embedded file name: /data/wduan/SOP-CG-master/src/controllers/dicg_controller.py
# Compiled at: 2023-06-06 14:50:33
import sys
from .basic_controller import BasicMAC
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional  as F
import numpy as np, contextlib, itertools, torch_scatter
from math import factorial
from random import randrange
from torch.distributions.relaxed_bernoulli import RelaxedBernoulli

from modules.agents import REGISTRY as agent_REGISTRY
from modules.action_encoders import REGISTRY as action_encoder_REGISTRY
from components.attention_module import AttentionModule
from components.gcn_module import GCNModule

import copy

class GroupMessageMAC(BasicMAC):

    def __init__(self, scheme, groups, args):
        super().__init__(scheme, groups, args)
        # self.residual = args.residual
        self.args = args
        self.n_gcn_layers = args.number_gcn_layers
        self.dicg_layers = []
        self.dicg_emb_hid = args.dicg_emb_hid
        # original input shape: obs + actions + agents : 10m vs 11m :105+17+10=132
        org_input_shape = self._get_input_shape(scheme) 

        self.gcn_message_dim = int( args.gcn_message_dim * scheme["obs"]["vshape"])
        print("gcn_message_dim",self.gcn_message_dim)

        self.concate_mlp_dim = args.concate_mlp_dim 
        agent_input_shape = org_input_shape
        if self.args.concate_gcn:
            agent_input_shape = agent_input_shape + self.gcn_message_dim
        if self.args.concate_gcn and self.args.concate_mlp:
            agent_input_shape = agent_input_shape + self.concate_mlp_dim
        # print("agent_input_shape",agent_input_shape)
        self._build_agents(agent_input_shape)

        self.mlp_emb_dim = org_input_shape
        self.mlp_encoder = self._mlp(org_input_shape, self.mlp_emb_dim, self.concate_mlp_dim)
        self.dicg_layers.append(self.mlp_encoder)
        self.attention_layer = AttentionModule((self.concate_mlp_dim), attention_type='general')
        self.dicg_layers.append(self.attention_layer)
        self.gcn_layers = nn.ModuleList([
                                GCNModule(in_features=(self.concate_mlp_dim), out_features=(self.gcn_message_dim), bias=True, id=0),
                                GCNModule(in_features=(self.gcn_message_dim), out_features=(self.gcn_message_dim), bias=True, id=1)
                                ])
        self.dicg_layers.extend(self.gcn_layers)
        # self.dicg_aggregator = self._mlp(self.gcn_message_dim, self.dicg_emb_hid, self.gcn_message_dim)

        self.temperature = 1
        self.adj_threshold = args.adj_threshold
        #-----------------Group devide---------------------------
        group_num = args.group_num
        self.trunk_size = args.obs_group_trunk_size
        self.group_in_shape = scheme["obs"]["vshape"] * self.trunk_size 
        self.groupnizer =  self._mlp(self.group_in_shape, self.mlp_emb_dim, group_num)
        self.small_eye_matrix =  0.001* torch.eye(self.n_agents).unsqueeze(0).cuda()

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        avail_actions = ep_batch['avail_actions'][:, t_ep]
        agent_outputs , _ , _ , _ = self.forward(ep_batch, t_ep, test_mode=test_mode)
        chosen_actions = self.action_selector.select_action((agent_outputs[bs]), (avail_actions[bs]), t_env, test_mode=test_mode)
        return chosen_actions

    def forward(self, ep_batch, t, test_mode=False):
        batch_size = ep_batch.batch_size
        obs_t = ep_batch["obs"][:, t]
        # if self.args.obs_cnn:
        #     org_agent_inputs, obs_emb = self.build_agent_cnn_inputs(ep_batch, t)
        #     # print("obs_emb",obs_emb.shape)
        # else:
        org_agent_inputs = self.build_agent_inputs(ep_batch, t)
        obs_mlp_emb = self.mlp_encoder.forward(org_agent_inputs) # [bs, num_agent,obs_dim] [1,64,200]



        #----------------------Group Mask---------------------------
        avail_actions = ep_batch['avail_actions'][:, t]
        
        embeddings_collection = []
        embeddings_collection.append(obs_mlp_emb)
        # print("embeddings_0",embeddings_0.shape)
        # attention_weights [10,10] [agent,agent]
        attention_weights = self.attention_layer.forward(obs_mlp_emb) #[batch_size, self.n_agents, self.n_agents]
        # print(attention_weights[0])
        # # sys.exit()
        #----------------------Group devide---------------------------
        group_index = None
        if t >= self.trunk_size:
            obs_trunk = ep_batch["obs"][:, t-self.trunk_size : t].permute(0, 2, 1, 3)
            # print("obs_trunk",obs_trunk.shape)
            group_index_temp = self.groupnizer(obs_trunk.reshape(batch_size,self.n_agents,-1))
            # print("group_index_temp",group_index_temp.shape)
            group_index = group_index_temp.softmax(dim=-1).argmax(dim=2) 
            # print("group_index",group_index.shape)
            # print(group_index[0])
            num_groups = len(torch.unique(group_index))
            group_mask = (group_index[:, :, None] == group_index[:, None, :]).float()
            # print(group_mask[0])
            covariance_matrix = torch.bmm(group_mask,group_mask.transpose(1, 2))
            PosDef_covariance_matrix = covariance_matrix + self.small_eye_matrix
            PosDef_min_value = torch.min(PosDef_covariance_matrix)
            PosDef_max_value = torch.max(PosDef_covariance_matrix)
            # Normalize the tensor to [0, 1]
            PosDef_covariance_matrix = (PosDef_covariance_matrix - PosDef_min_value) / (PosDef_max_value - PosDef_min_value)
            # print(PosDef_covariance_matrix[0])
            PosDef_covariance_matrix = PosDef_covariance_matrix.unsqueeze(1).repeat(1, self.n_agents, 1, 1)
            mvn1 = torch.distributions.MultivariateNormal(attention_weights, covariance_matrix=PosDef_covariance_matrix)
            samples = mvn1.sample((1,))
            final_graph = samples[0]
            final_graph = 0.5 * (final_graph + final_graph.transpose(-1, -2))

            # Normalize the tensor to [0, 1]
            min_value = torch.min(final_graph)
            max_value = torch.max(final_graph)
            final_graph = (final_graph - min_value) / (max_value - min_value)
            # print(final_graph[0])
            if self.args.is_sparse:
                    final_graph = (final_graph > self.adj_threshold).float() * final_graph
            # print(final_graph[0])
            # sys.exit()
        #----------------------Group devide---------------------------
        else:
            attention_dist = RelaxedBernoulli(self.temperature, logits=attention_weights.view(ep_batch.batch_size, -1))
            adj_sample = attention_dist.sample().view(ep_batch.batch_size, self.n_agents, self.n_agents)
            adj_sample = 0.5 * (adj_sample + adj_sample.transpose(-1, -2))
            final_graph = (adj_sample > self.adj_threshold).float() * adj_sample
            # print(final_graph[0])


        # print("result_matrix", result_matrix.shape)
        for i_layer, gcn_layer in enumerate(self.gcn_layers):
            embeddings_gcn = gcn_layer.forward(embeddings_collection[i_layer], final_graph)
            # print("embeddings_gcn",embeddings_gcn.shape)
            embeddings_collection.append(embeddings_gcn)
        
        if self.args.concate_gcn and self.args.concate_mlp:
            temp_org_input = org_agent_inputs.view(-1,org_agent_inputs.shape[-1])
            # print("temp_org_input",temp_org_input.shape)
            temp_mlp_message = embeddings_collection[0].view(-1,self.concate_mlp_dim)
            # print("temp_mlp_message",temp_mlp_message.shape)
            temp_gcn_message = embeddings_collection[-1].view(-1,self.gcn_message_dim)
            # print("temp_gcn_message",temp_gcn_message.shape)
            agent_input = th.cat([temp_org_input,temp_mlp_message, temp_gcn_message], dim=1)
        elif self.args.concate_gcn:
            temp_org_input = org_agent_inputs.view(-1,org_agent_inputs.shape[-1])
            # print("temp_org_input",temp_org_input.shape)
            temp_gcn_message = embeddings_collection[-1].view(-1,self.gcn_message_dim)
            # print("temp_gcn_message",temp_gcn_message.shape)
            agent_input = th.cat([temp_org_input, temp_gcn_message], dim=1)
        else:
            agent_input = org_agent_inputs.view(-1,org_agent_inputs.shape[-1])
        # print("agent_input",agent_input.shape)

        # dicg_agent_inputs = self.dicg_aggregator.forward(dicg_agent_inputs)
        agent_outs, self.hidden_states = self.agent(agent_input, self.hidden_states)
        if self.agent_output_type == 'pi_logits':
            if getattr(self.args, 'mask_before_softmax', True):
                reshaped_avail_actions = avail_actions.reshape(batch_size * self.n_agents, -1)
                agent_outs[reshaped_avail_actions == 0] = -10000000000.0
            agent_outs = th.nn.functional.softmax(agent_outs, dim=(-1))
            # if not test_mode:
            #     epsilon_action_num = agent_outs.size(-1)
            #     if getattr(self.args, 'mask_before_softmax', True):
            #         epsilon_action_num = reshaped_avail_actions.sum(dim=1, keepdim=True).float()
            #     agent_outs = (1 - self.action_selector.epsilon) * agent_outs + th.ones_like(agent_outs) * self.action_selector.epsilon / epsilon_action_num
            #     if getattr(self.args, 'mask_before_softmax', True):
            #         agent_outs[reshaped_avail_actions == 0] = 0.0
        # sys.exit()
        return agent_outs.view(batch_size, self.n_agents, -1), torch.cat((attention_weights,final_graph),2), group_index , obs_mlp_emb
        
    def build_agent_inputs(self, batch, t):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t])  # b1av
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t-1])
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))
        inputs = th.cat([x.reshape(bs, self.n_agents, -1) for x in inputs], dim=2)
        return inputs

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            # print("input_shape",input_shape)
            # print(scheme["actions_onehot"]["vshape"][0])
            input_shape += scheme["actions_onehot"]["vshape"][0]

        if self.args.obs_agent_id:
            input_shape += self.n_agents

        return input_shape

    # encode adj to onehot matrix    
    # def encode_onehot(self, labels):
    #     classes = set(labels)
    #     classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    #     labels_onehot = np.array(list(map(classes_dict.get, labels)),dtype=np.int32)
    #     return labels_onehot

    def parameters(self):
        param = itertools.chain(BasicMAC.parameters(self), self.mlp_encoder.parameters(), self.attention_layer.parameters(), self.gcn_layers.parameters())
        return param

    def load_state(self, other_mac):
        BasicMAC.load_state(self, other_mac)
        self.mlp_encoder.load_state_dict(other_mac.mlp_encoder.state_dict())
        self.attention_layer.load_state_dict(other_mac.attention_layer.state_dict())
        self.gcn_layers.load_state_dict(other_mac.gcn_layers.state_dict())

        # self.dicg_aggregator.load_state_dict(other_mac.dicg_aggregator.state_dict())

    def cuda(self):
        self.agent.cuda()
        self.mlp_encoder.cuda()
        self.attention_layer.cuda()
        self.gcn_layers.cuda()
        self.groupnizer.cuda()
        # self.dicg_aggregator.cuda()

    def save_models(self, path):
        # BasicMAC.save(self, path)
        th.save(self.mlp_encoder.state_dict(), '{}/mlp_encoder.th'.format(path))
        th.save(self.attention_layer.state_dict(), '{}/attention_layer.th'.format(path))
        th.save(self.gcn_layers.state_dict(), '{}/gcn_layers.th'.format(path))
        # th.save(self.dicg_aggregator.state_dict(), '{}/dicg_aggregator.th'.format(path))

    def load_models(self, path):
        # BasicMAC.load_state_dict(self, path)
        self.mlp_encoder.load_state_dict(th.load(('{}/mlp_encoder.th'.format(path)), map_location=(lambda storage, loc: storage)))
        self.attention_layer.load_state_dict(th.load(('{}/attention_layer.th'.format(path)), map_location=(lambda storage, loc: storage)))
        self.gcn_layers.load_state_dict(th.load(('{}/gcn_layers.th'.format(path)), map_location=(lambda storage, loc: storage)))

        # self.dicg_aggregator.load_state_dict(th.load(('{}/dicg_aggregator.th'.format(path)), map_location=(lambda storage, loc: storage)))

    def _build_agents(self, input_shape):
        self.agent = agent_REGISTRY[self.args.agent](input_shape, self.args)

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
    