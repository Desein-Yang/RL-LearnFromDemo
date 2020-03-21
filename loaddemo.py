import pickle
import numpy as np

"""
datastructure 
demo dict
- actions
- signed rewards
- reward
- lives
- checkpoints
- checkpoint_action_nr
"""

class DemoReplay:
    def __init__(self,demo_file_name):
        assert demo_file_name is not None
        with open(demo_file_name, "rb") as f:
            dat = pickle.load(f)
        self.actions = dat['actions']
        self.rewards = dat['rewards']
        self.returns = np.cumsum(self.rewards)
        assert len(self.rewards) == len(self.actions)
        self.lives = dat['lives']

        self.checkpoints = dat['checkpoints']
        self.checkpoint_action_nr = dat['checkpoint_action_nr']
        self.starting_point = len(self.actions) - 1 - seed//workers_per_sp
        self.starting_point_current_ep = None  

        #TODO:what are seed and workers per sp          


class ReplayResetEnv():
    """
        Randomly resets to states from a replay
    """
