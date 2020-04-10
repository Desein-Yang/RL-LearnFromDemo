# NOTE this is not my code, code was taken from: https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py

import tempfile
import os
import random
import pickle
import gym
import cv2
import imageio
import numpy as np

from collections import deque
from PIL import Image
from gym import spaces
from multiprocessing import Process, Pipe
from collections import deque
from PIL import Image
from gym.wrappers import TimeLimit

reset_for_batch = False
cv2.ocl.setUseOpenCL(False)
os.environ.setdefault('PATH', '')

# Atari preprocessing
# env = gym.make('PongNoFrameskip-v4')
# env = AtariPreprocessing(env, screen_size=84, grayscale_obs=grayscale, scale_obs=scaled,frame_skip=4, noop_max=30)
class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self, **kwargs):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.randint(1, self.noop_max + 1) #pylint: disable=E1101
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)

class FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        """Take action on reset for environments that are fixed until firing."""
        gym.Wrapper.__init__(self, env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)

class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done  = True

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # for Qbert sometimes we stay in lives == 0 condition for a few frames
            # so it's important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, info

    def reset(self, **kwargs):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs

class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros((2,)+env.observation_space.shape, dtype=np.uint8)
        self._skip       = skip

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2: self._obs_buffer[0] = obs
            if i == self._skip - 1: self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

class ClipRewardEnv(gym.RewardWrapper):
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)

    def reward(self, reward):
        """Bin reward to {+1, 0, -1} by its sign."""
        return np.sign(reward)

class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env, width=84, height=84, grayscale=True, dict_space_key=None):
        """
        Warp frames to 84x84 as done in the Nature paper and later work.
        If the environment uses dictionary observations, `dict_space_key` can be specified which indicates which
        observation should be warped.
        """
        super().__init__(env)
        self._width = width
        self._height = height
        self._grayscale = grayscale
        self._key = dict_space_key
        if self._grayscale:
            num_colors = 1
        else:
            num_colors = 3

        new_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self._height, self._width, num_colors),
            dtype=np.uint8,
        )
        if self._key is None:
            original_space = self.observation_space
            self.observation_space = new_space
        else:
            original_space = self.observation_space.spaces[self._key]
            self.observation_space.spaces[self._key] = new_space
        assert original_space.dtype == np.uint8 and len(original_space.shape) == 3

    def observation(self, obs):
        if self._key is None:
            frame = obs
        else:
            frame = obs[self._key]

        if self._grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(
            frame, (self._width, self._height), interpolation=cv2.INTER_AREA
        )
        if self._grayscale:
            frame = np.expand_dims(frame, -1)

        if self._key is None:
            obs = frame
        else:
            obs = obs.copy()
            obs[self._key] = frame
        return obs

class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        """Stack k last frames.
        Returns lazy array, which is much more memory efficient.
        See Also
        --------
        baselines.common.atari_wrappers.LazyFrames
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(low=0, high=255, shape=(shp[:-1] + (shp[-1] * k,)), dtype=env.observation_space.dtype)

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))

class ScaledFloatFrame(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=env.observation_space.shape, dtype=np.float32)

    def observation(self, observation):
        # careful! This undoes the memory optimization, use
        # with smaller replay buffers only.
        return np.array(observation).astype(np.float32) / 255.0

class LazyFrames(object):
    def __init__(self, frames):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.
        This object should only be converted to numpy array before being passed to the model.
        You'd not believe how complex the previous solution was."""
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=-1)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]

    def count(self):
        frames = self._force()
        return frames.shape[frames.ndim - 1]

    def frame(self, i):
        return self._force()[..., i]

class ReplayResetEnv(gym.Wrapper):
    """
        Randomly resets to states from a replay
    """
    def __init__(self,env,demo,allowed_lag=50,allowed_score_deficit=0):
        gym.Wrapper.__init__(self, env)
        self.env = env 
        self.demo = demo
        self.score = 0
        self.cur_demo_idx = 0
        self.actions_to_overwrite = []
        self.action_nr = -1
        self.start_point = -1

        self.allowed_lag = allowed_lag
        self.allowed_score_deficit = allowed_score_deficit
        

    def step(self,action):
        # If there is actions in cache need to overwrite  
        # it means the env will perform action in list to reset to reset point
        if len(self.actions_to_overwrite) > 0:
            action = self.actions_to_overwrite.pop(0)
            valid = False
        else:
            valid = True
        prev_lives = self.env.unwrapped.ale.lives()
        obs, reward, done, info = self.env.step(action)
        

        self.action_nr += 1
        self.score += reward

        # kill if we have achieved the final score, or if we're laggging the demo too much
        if self.score >= self.demo.returns[-1]:
            done = True
            info['achieved_done'] = True # distinguish kill done or done
        
        # Because randomness we may have more action when we reset
        # so we set an alllowed deficit to control
        if self.action_nr > self.allowed_lag:
            # min index and max index is range of episode action_nr
            min_index = max(self.action_nr - self.allowed_lag,0)
            max_index = self.action_nr + self.allowed_lag
            if self.score < min(self.demo.returns[min_index:max_index]) - self.allowed_score_deficit:
                done = True

        # write info
        if self.action_nr < self.start_point + 100:
            info['increase_entropy'] = True
        
        if done:
            info['episode_info'] = {
                'good_as_demo':self.score >= self.demo.returns[-1],
                'action_nr':self.action_nr,
                'live':self.env.unwrapped.ale.lives(),
                'score':self.score,
                'valid':valid
            }
        return obs, reward, done, info

    # when reset is None we perform default reset
    # when reset is int we reset to specific point 
    def reset(self,reset_point=None,start_point=None):
        obs = self.env.reset()

        if reset_point == None or start_point == None:
            self.cur_demo_idx = 0
            self.action_nr = -1
            self.score = 0
            self.actions_to_overwrite = []
            self.start_point = -1
            ob = self.env.reset()
            # TODO: do we need noops here?
            noops = random.randint(0,30)
            for _ in range(noops):
                obs, _, _, _ = self.env.step(0)
            return obs
        
        elif reset_point > 0 and reset_point < self.demo.length: 
            # action before reset point need to be perform
            self.start_point = start_point
            self.action_nr = 0
            self.score = self.demo.returns[reset_point]
            self.actions_to_overwrite = self.demo.actions[:reset_point]
            for nr,ob in zip(self.demo.action_nr[::-1],self.demo.obs[::-1]):
                if nr <= self.start_point:
                    start_nr = nr
                    start_ob = ob
                    break
                if start_nr > 0:
                    self.env.unwrapped.restore_state(start_ob)
                start_nr_lstm = np.maximum(self.start_point,start_nr)
                if start_nr_lstm > start_nr:
                    for a in self.demo.actions[start_nr:start_nr_lstm]:
                        action = self.env.unwrapped._action_set[a]
                        self.env.wrapped.ale.act(action)
                self.actions_to_overwrite = self.demo.actions[start_nr_lstm:self.start_point]
                if start_nr_lstm > 0:
                    obs = self.env.unwrapped._get_image()
                self.action_nr = start_nr_lstm
                self.score = self.demo.returns[start_nr_lstm]
                if self.start_point == 0 and self.actions_to_overwrite == []:
                    noops = random.randint(0, 30)
                    for _ in range(noops):
                        obs, _, _, _ = self.env.step(0)
            return obs
                

    def move_start_point(self,delta):
        self.start_point = int(np.maximum(self.start_point-delta,0))

# class ReplayResetEnv1():
#     """
#         Randomly resets to states from a replay
#     """
#     def __init__(self,
#                  env,
#                  demo_file_name,
#                  seed,
#                  reset_steps_ignored=64,
#                  workers_per_sp=4,
#                  frac_sample=0.2,
#                  game_over_on_life_loss=True,
#                  allowed_lag=50,
#                  allowed_score_deficit=0,
#                  test_from_start=False):
#         super(ReplayResetEnv, self).__init__(env)
#         self.rng = np.random.RandomState(seed)
#         self.reset_steps_ignored = reset_steps_ignored
#         self.actions_to_overwrite = [] # 重写action
#         self.frac_sample = frac_sample
#         self.game_over_on_life_loss = game_over_on_life_loss # 标志量一丢命就结束
#         self.allowed_lag = allowed_lag
#         self.allowed_score_deficit = allowed_score_deficit
#         self.demo_replay_info = [] #
#         self.test_from_start = test_from_start
#         if test_from_start:
#             self.demo_replay_info.append(DemoReplayInfo(None, seed, workers_per_sp))
#         if os.path.isdir(demo_file_name):
#             import glob
#             for f in sorted(glob.glob(demo_file_name + '/*.demo')):
#                 self.demo_replay_info.append(DemoReplayInfo(f, seed, workers_per_sp))
#         else:
#             self.demo_replay_info.append(DemoReplayInfo(demo_file_name, seed, workers_per_sp))
#         self.cur_demo_replay = None # 待回放的demo
#         self.cur_demo_idx = -1 # 当前进行的在demo中的idx
#         self.extra_frames_counter = -1 # 帧计数器
#         self.action_nr = -1
#         self.score = -1

#     def recursive_getattr(self, name):
#         prefix = 'starting_point_'
#         if name[:len(prefix)] == prefix:
#             idx = int(name[len(prefix):])
#             return self.demo_replay_info[idx].starting_point
#         elif name == 'n_demos':
#             return len(self.demo_replay_info)
#         else:
#             return super(ReplayResetEnv, self).recursive_getattr(name)

#     def step(self, action):
#         if len(self.actions_to_overwrite) > 0:
#             action = self.actions_to_overwrite.pop(0)
#             valid = False
#         else:
#             valid = True
#         prev_lives = self.env.unwrapped.ale.lives()
#         obs, reward, done, info = self.env.step(action)
#         info['idx'] = self.cur_demo_idx
#         self.action_nr += 1
#         self.score += reward

#         # game over on loss of life, to speed up learning
#         if self.game_over_on_life_loss:
#             lives = self.env.unwrapped.ale.lives()
#             if lives < prev_lives and lives > 0:
#                 done = True

#         if self.test_from_start and self.cur_demo_idx == 0:
#             pass
#         # kill if we have achieved the final score, or if we're laggging the demo too much
#         elif self.score >= self.cur_demo_replay.returns[-1]:
#             self.extra_frames_counter -= 1
#             if self.extra_frames_counter <= 0:
#                 done = True
#                 info['replay_reset.random_reset'] = True # to distinguish from actual game over
#         elif self.action_nr > self.allowed_lag:
#             min_index = self.action_nr - self.allowed_lag
#             if min_index < 0:
#                 min_index = 0
#             if min_index >= len(self.cur_demo_replay.returns):
#                 min_index = len(self.cur_demo_replay.returns) - 1
#             max_index = self.action_nr + self.allowed_lag
#             threshold = min(self.cur_demo_replay.returns[min_index: max_index]) - self.allowed_score_deficit
#             # 允许的赤字
#             if self.score < threshold:
#                 done = True

#         # output flag to increase entropy if near the starting point of this episode
#         if self.action_nr < self.cur_demo_replay.starting_point + 100:
#             info['increase_entropy'] = True

#         if done:
#             ep_info = {'l': self.action_nr,
#                        'as_good_as_demo': (self.score >= (self.cur_demo_replay.returns[-1] - self.allowed_score_deficit)),
#                        'r': self.score,
#                        'starting_point': self.cur_demo_replay.starting_point_current_ep,
#                        'idx': self.cur_demo_idx}
#             info['episode'] = ep_info

#         if not valid:
#             info['replay_reset.invalid_transition'] = True

#         return obs, reward, done, info

#     def decrement_starting_point(self, nr_steps, demo_idx):
#         if self.demo_replay_info[demo_idx].starting_point>0:
#             self.demo_replay_info[demo_idx].starting_point = int(np.maximum(self.demo_replay_info[demo_idx].starting_point - nr_steps, 0))

#     def reset(self):
#         obs = self.env.reset()
#         self.extra_frames_counter = int(np.exp(self.rng.rand()*7))

#         self.cur_demo_idx = random.randint(0, len(self.demo_replay_info) - 1)
#         self.cur_demo_replay = self.demo_replay_info[self.cur_demo_idx]

#         if self.test_from_start and self.cur_demo_idx == 0:
#             self.cur_demo_replay.starting_point_current_ep = 0
#             self.actions_to_overwrite = []
#             self.action_nr = 0
#             self.score = 0
#             obs = self.env.reset()
#             noops = random.randint(0, 30)
#             for _ in range(noops):
#                 obs, _, _, _ = self.env.step(0)
#             return obs

#         elif reset_for_batch:
#             self.cur_demo_replay.starting_point_current_ep = 0
#             self.actions_to_overwrite = self.cur_demo_replay.actions[:]
#             self.action_nr = 0
#             self.score = self.cur_demo_replay.returns[0]
#         else:
#             if self.rng.rand() <= 1.-self.frac_sample:
#                 self.cur_demo_replay.starting_point_current_ep = self.cur_demo_replay.starting_point
#             else:
#                 self.cur_demo_replay.starting_point_current_ep = self.rng.randint(low=self.cur_demo_replay.starting_point, high=len(self.cur_demo_replay.actions))

#             start_action_nr = 0
#             start_ckpt = None
#             for nr, ckpt in zip(self.cur_demo_replay.checkpoint_action_nr[::-1], self.cur_demo_replay.checkpoints[::-1]):
#                 if nr <= (self.cur_demo_replay.starting_point_current_ep - self.reset_steps_ignored):
#                     start_action_nr = nr
#                     start_ckpt = ckpt
#                     break
#             if start_action_nr > 0:
#                 self.env.unwrapped.restore_state(start_ckpt)
#             nr_to_start_lstm = np.maximum(self.cur_demo_replay.starting_point_current_ep - self.reset_steps_ignored, start_action_nr)
#             if nr_to_start_lstm>start_action_nr:
#                 for a in self.cur_demo_replay.actions[start_action_nr:nr_to_start_lstm]:
#                     action = self.env.unwrapped._action_set[a]
#                     self.env.unwrapped.ale.act(action)
#             self.cur_demo_replay.actions_to_overwrite = self.cur_demo_replay.actions[nr_to_start_lstm:self.cur_demo_replay.starting_point_current_ep]
#             if nr_to_start_lstm>0:
#                 obs = self.env.unwrapped._get_image()
#             self.action_nr = nr_to_start_lstm
#             self.score = self.cur_demo_replay.returns[nr_to_start_lstm]
#             if self.cur_demo_replay.starting_point_current_ep == 0 and self.cur_demo_replay.actions_to_overwrite == []:
#                 noops = random.randint(0, 30)
#                 for _ in range(noops):
#                     obs, _, _, _ = self.env.step(0)

#         return obs

def build_env(env_id,seed,rank=0,max_episode_steps=None):
    """build envrionment with preprocess as DQN"""
    env = gym.make(env_id)
    assert 'NoFrameskip' in env.spec.id
    env = NoopResetEnv(env, noop_max=30) # NoopResetEnv为gym.Wrapper的继承类。每次环境重置（调用reset()）时执行指定步随机动作。
    env = MaxAndSkipEnv(env, skip=4) # MaxAndSkipEnd也是gym.Wrapper的继承类。每隔4帧返回一次。返回中的reward为这４帧reward
    if max_episode_steps is not None:
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
    env.seed(seed + rank)
    return env

def wrap_deepmind(env, episode_life=True, clip_rewards=True, frame_stack=False, scale=False):
    """Configure environment for DeepMind-style Atari.
    """
    if episode_life:
        env = EpisodicLifeEnv(env)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = WarpFrame(env)
    if scale:
        env = ScaledFloatFrame(env)
    if clip_rewards:
        env = ClipRewardEnv(env)
    if frame_stack:
        env = FrameStack(env, 4)
    return env

def make_atari_env(env_id,num_env,seed):
    return [build_env(env_id,seed,i) for i in range(num_env)]


    

