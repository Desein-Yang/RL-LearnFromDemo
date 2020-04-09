import gym
import os,sys
from env import build_env,ReplayResetEnv
from loaddemo import Demo,Buffer

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

    agent = RandomAgent(env.action_space)
    episode_count = 100
    reward = 0
    done = False

    for i in range(episode_count):
        ob = env.reset(10)
        print("successful reset to point")
        for i in range(3):
            action = agent.act(ob, reward, done)
            ob, reward, done, info = env.step(action)
            print("successful step")
            if done:
                print("successful done")
                break
        action, ob, reward, done, info = demo.replay(1)
        buffer = Buffer()
        buffer.add(action,ob,reward,done,info,False)
    env.close()
