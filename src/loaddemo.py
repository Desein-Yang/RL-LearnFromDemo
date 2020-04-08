import pickle
import numpy as np

"""
datastructure 
demo dict
- actions
- signed rewards
- reward
- lives
- checkpoints (observations)
- checkpoint_action_nr
demo info
- idx              index in demo
- increase_entropy flag
- as good as demo  flag
- action_nr        action_nr
- valid            flag
- live             live
- score            return
"""

class Demo:
    def __init__(self,demo_file_name):
        assert demo_file_name is not None
        if demo_file_name is None:
            self.actions = None
            self.rewards = None
            self.returns = [0]
            self.obs = None
            self.lives = None
            self.checkpoint_action_nr = None
        else:
            with open(demo_file_name, "rb") as f:
                dat = pickle.load(f)
                self.actions = dat['actions']
                self.rewards = dat['rewards']
                self.length = len(self.actions)
                for i in range(self.length):
                    self.returns.append(np.cumsum(self.rewards[:i]))
                assert len(self.rewards) == len(self.actions)
                self.lives = dat['lives']
                self.obs = dat['checkpoints']
                self.checkpoint_action_nr = dat['checkpoint_action_nr']

    def replay(self,i):
        """return action, ob, reward, done, info in demo.  
        info??"""
        info = {}
        if self.lives[i] == 0:
            done = True
        else:
            done = False
        info['episode_info']={
            'live':self.lives[i],
            'score':self.returns[i],
            'action_nr':self.checkpoint_action_nr[i],
        }
        info['idx'] = i
        info['achived_done'] = False
        return self.actions[i],self.obs[i],self.rewards[i],done,info
                                              

if __name__ == "__main__":
    demo = Demo('Phase2\demos\Pitfall.demo')
    print(demo)
