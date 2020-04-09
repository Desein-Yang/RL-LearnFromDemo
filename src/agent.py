import logging
import os
import sys
import gym
import random
import numpy as np
import torch
import time
import copy
import math
# from nn_builder.pytorch.NN import NN
from model import CNNModel,MLPModel
from torch.optim import SGD
from torch import nn
from torch.distributions import Categorical,Normal
import torch.nn.functional as F
from collections import deque
from loaddemo import Buffer

class Worker(object):
    """worker for rollout in enviroment"""
    def __init__(self,env,ARGS):
        self.env = env   
        self.actor_model = MLPModel(ARGS.FRAME_SKIP,ARGS.action_n,True)
        self.critic_model = MLPModel(ARGS.FRAME_SKIP,1,False)
        self.reset_point = ARGS.T
        self.timestep_limit = ARGS.timestep_limit
        self.gamma = ARGS.gamma
        self.eps = ARGS.epsilon

    def act(self,ob):
        ob = torch.from_numpy(ob) # ob is a numpy array
        action_prob = self.actor_model(ob)
        dist = Categorical(action_prob)
        # dist = Normal(action_prob)
        action = dist.sample()
        # ac_log_prob = dist.log_prob(action)
        return action

    def update(self,actor_model,critic_model,reset_point):
        self.actor_model.load_state_dict(actor_model)
        self.critic_model.load_state_dict(critic_model)
        self.reset_point = reset_point

    def set_random_seeds(self, random_seed):
        """Sets all possible random seeds so results can be reproduced"""
        os.environ['PYTHONHASHSEED'] = str(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(random_seed)
        # tf.set_random_seed(random_seed)
        random.seed(random_seed)
        np.random.seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_seed)
            torch.cuda.manual_seed(random_seed)
        if hasattr(gym.spaces, 'prng'):
            gym.spaces.prng.seed(random_seed)

    def rollout(self,demo,ARGS):
        """Rollout worker for evaluation and return batch D and success times W"""
        # TODO: reset to state function
        # get action()
        # replay demo()
        L = ARGS.batch_rollout_size
        K = ARGS.rnn_memory_size
        d = ARGS.nums_start_point
        W = 0 # success rate W 
        D = Buffer()

        reset_point = self.reset_point
        start_point = np.random.uniform(reset_point - d,reset_point) # start = tao * reset = tao
        self.env.reset_to_state(reset_point)
        i = start_point - K

        for step in range(L):
            if i > start_point:
                # sample action
                action = self.act(ob)
                ob, reward, done, info = self.env.step(action)
                is_train = True
            else:
                action, ob, reward, done, info = demo.replay(i)
                is_train = False
            D.add(action,ob,reward,done,info,is_train)
            i = i + 1

            if done:
                if sum(D.rewards[reset_point:]) > sum(demo.rewards[reset_point:]):
                    W = W + 1
                start_point = np.random.uniform(reset_point-d,reset_point)
                i = start_point - K
                self.env.reset_to_state(reset_point)      
        
        return D,W

class PPO_Optimizer(object):
    def __init__(self,ARGS):
        # ob_space = self.env.observation_space
        # ac_space = self.env.action_space

        # ARGS.action_n = self.env.action_space.n
        self.adv = []
        self.eps = ARGS.eps
        self.w_vf = ARGS.weight_vf
        self.w_ent = ARGS.weight_entropy
        self.gamma = ARGS.gamma
        self._max_grad_norm = ARGS.max_grad_norm

        self.actor_old = MLPModel(ARGS.FRAME_SKIP,ARGS.action_n,True)
        self.critic_old = MLPModel(ARGS.FRAME_SKIP,1,False)
        self.actor = MLPModel(ARGS.FRAME_SKIP,ARGS.action_n,True)
        self.critic = MLPModel(ARGS.FRAME_SKIP,1,False)
        self.actor_optim = SGD(self.actor.parameters(), lr=ARGS.lr)
        self.critic_optim = SGD(self.critic.parameters(), lr=ARGS.lr)

    # Finished
    def get_advantage_est(self,buffer,model = 'old'):
        """get advantage value estimation as equation (10)"""
        adv = []
        for ob in buffer.obs:
            if model == 'new':
                buffer.state_value.append(self.critic(torch.from_numpy(ob)))
            elif model == 'old':
                buffer.state_value.append(self.critic_old(torch.from_numpy(ob)))
            else:
                raise OSError        
        assert len(buffer.log_prob) == len(buffer.discount_r)
        for state_value,d_reward in zip(buffer.state_value,buffer.discount_r):
            adv.append(d_reward - state_value)
        return adv

    # Finished
    def get_discounted_reward(self,buffer,norm=True):
        """return [R[0],R[1]...R[n]]"""
        discount_r = []
        value = self.critic(torch.from_numpy(buffer.obs[-1]))
        for reward in buffer.rewards:
            discount_r.insert(0,value)
            value += value * self.gamma + reward
        discount_r = torch.tensor(discount_r)
        if norm:
            discount_r = (discount_r-discount_r.mean())/(discount_r.std()+self.eps)
        buffer.discount_r = discount_r
        return discount_r

    def optimize(self,actor_model,critic_model,D):
        losses, clip_losses, vf_losses, ent_losses = [], [], [], []
        self.actor_old.load_state_dict(self.actor.state_dict())
        self.critic_old.load_state_dict(self.critic.state_dict())
        self.actor.load_state_dict(actor_model.state_dict())
        self.critic.load_state_dict(critic_model.state_dict()) 
        
        self.get_discounted_reward(D)
        adv = self.get_advantage_est(D)
        action = torch.Tensor(D.action)
        for idx in range(D.size):
            ob = torch.from_numpy(D.obs[idx])
            dist = Categorical(self.actor(ob))
            dist_old = Categorical(self.actor_old(ob))
            ratio = torch.exp(dist.log_prob(action)-dist_old.log_prob(action))
            surr1 = ratio * adv
            surr2 = ratio.clamp(1. - self.eps, 1. + self.eps ) * adv

            clip_loss = - torch.min(surr1,surr2).mean()
            clip_losses.append(clip_loss.numpy())

            vf_loss = F.smooth_l1_loss(self.critic(ob),D.state_value)
            vf_losses.append(vf_loss.numpy())

            ent_loss = dist.entropy().mean()
            ent_losses.append(ent_loss.numpy())

            loss = clip_loss + self.w_vf * vf_loss + self.w_ent * ent_loss
            losses.append(loss.numpy())

            self.actor_optim.zero_grad()
            self.critic_optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(list(
                self.actor.parameters()) + list(self.critic.parameters()),
                                        self._max_grad_norm)

            # update parameter of actor and critic
            self.actor_optim.step()
            self.critic_optim.step()

        lossdict = {
            'loss' : losses,
            'Cliploss' : clip_losses,
            'VF  loss' : vf_losses,
            'Ent loss' : ent_losses,
        }
        return lossdict, self.actor.state_dict(), self.critic.state_dict()
