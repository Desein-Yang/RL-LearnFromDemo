import gym
import os,sys
from env import build_env,ReplayResetEnv
from loaddemo import Demo,Buffer
from agent import Worker,PPO_Optimizer
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class ARGS:
    gamename = 'AlienDeterministic'
    M = 10 # M parallel worker
    delta = 10 # delta start point shift size 
    rho = 0.1 # success threhold rho
    T = 1024 # demostration length T
    algorithm = "PPO" # laerning algorithm
    logger = None

    eps = 0.2 # PPO eps
    lr = 0.1 # PPO learning rate
    
    ncpu = 1
    timestep_limit = 1000000
    action_n = 0
    
    # fixed
    seed = 111
    FRAME_SKIP = 4
    gamma = 0.99
    epsilon = 1e-8
    allowed_lag = 50
    allowed_score_deficit = 0
    max_grad_norm = 1
    weight_vf = 0.5
    weight_entropy = 1e-4

    @classmethod
    def set_params(cls,kwargs):
        cls.gamename = kwargs.game+'Deterministic'
        cls.M = kwargs.worker_num
        cls.delta = kwargs.shift_size
        cls.rho = kwargs.success_threhold
        cls.T = kwargs.demo_len
        cls.ncpu = kwargs.ncpu
    
    @classmethod
    def set_logger(cls, logger):
        cls.logger = logger

    @classmethod
    def set_folder_path(cls,folder_path):
        cls.folder_path = folder_path

    @classmethod
    def set_env(cls,env):
        cls.action_n = env.action_space.n
    
    @classmethod
    def output(cls):
        logger = cls.logger
        logger.info("Game:%s" % cls.gamename)
        logger.info("Worker number%s" % cls.M)
        # logger.info("")

class RandomAgent(object):
    """The world's simplest agent!"""
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()

# successful test ReplayDemoEnv(),Buffer(),Demo(),and build_env()
if __name__ == "__main__":
    demo = Demo('demos/Pitfall.demo')
    env = build_env('AlienNoFrameskip-v4',111)
    env = ReplayResetEnv(env,demo,50,0)
    print("Env is set")

    env.reset()
    print("Successful reset")

    # agent = RandomAgent(env.action_space)
    agent = Worker(env,ARGS)
    optimizer = PPO_Optimizer(ARGS)
    print("agent and optimizer are set")
    episode_count = 100
    reward = 0
    done = False

    for i in range(episode_count):
        ob = env.reset(10)
        print("successful reset to point")
        for i in range(3):
            action = agent.act(ob)
            ob, reward, done, info = env.step(action)
            print("successful step")
            if done:
                print("successful done")
                break
        #action, ob, reward, done, info = demo.replay(1)
        #buffer = Buffer()
        #buffer.add(action,ob,reward,done,info,False)
    D,W = agent.rollout(demo,ARGS)
    env.close()
