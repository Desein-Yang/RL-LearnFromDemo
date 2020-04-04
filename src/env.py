# NOTE this is not my code, code was taken from: https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py

os.environ.setdefault('PATH', '')
cv2.ocl.setUseOpenCL(False)

import tempfile
import os
import random
import pickle
import gym
from collections import deque
from PIL import Image
from gym import spaces
import imageio
import numpy as np
from multiprocessing import Process, Pipe
mpi4py.rc.initialize = False
import cv2
reset_for_batch = False


class MyWrapper(gym.Wrapper):
    def __init__(self, env):
        super(MyWrapper, self).__init__(env)

    def decrement_starting_point(self, nr_steps, idx):
        return self.env.decrement_starting_point(nr_steps, idx)

    def recursive_getattr(self, name):
        if hasattr(self, name):
            return getattr(self, name)
        else:
            try:
                return self.env.recursive_getattr(name)
            except AttributeError:
                raise Exception(f'Couldn\'t get attr: {name}')

    def batch_reset(self):
        global reset_for_batch
        reset_for_batch = True
        obs = self.env.reset()
        reset_for_batch = False
        return obs

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def step_async(self, actions):
        return self.env.step_async(actions)

    def step_wait(self):
        return self.env.step_wait()

    def reset_task(self):
        return self.env.reset_task()

    @property
    def num_envs(self):
        return self.env.num_envs

class VecFrameStack(MyWrapper):
    """
    Vectorized environment base class
    """
    def __init__(self, venv, nstack):
        self.venv = venv
        self.nstack = nstack
        wos = venv.observation_space # wrapped ob space
        low = np.repeat(wos.low, self.nstack, axis=-1)
        high = np.repeat(wos.high, self.nstack, axis=-1)
        self.stackedobs = np.zeros((venv.num_envs,)+low.shape, low.dtype)
        self._observation_space = spaces.Box(low=low, high=high)
        self._action_space = venv.action_space

    def step(self, vac):
        """
        Apply sequence of actions to sequence of environments
        actions -> (observations, rewards, news)
        where 'news' is a boolean vector indicating whether each element is new.
        """
        obs, rews, news, infos = self.venv.step(vac)
        self.stackedobs = np.roll(self.stackedobs, shift=-1, axis=-1)
        for (i, new) in enumerate(news):
            if new:
                self.stackedobs[i] = 0
        self.stackedobs[..., -obs.shape[-1]:] = obs
        return self.stackedobs, rews, news, infos

    def reset(self):
        """
        Reset all environments
        """
        obs = self.venv.reset()
        self.stackedobs[...] = 0
        self.stackedobs[..., -obs.shape[-1]:] = obs
        return self.stackedobs

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._observation_space

    def close(self):
        self.venv.close()

    @property
    def num_envs(self):
        return self.venv.num_envs

class MaxAndSkipEnv(MyWrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        MyWrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = deque(maxlen=2)
        self._skip       = skip

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        combined_info = {}
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            combined_info.update(info)
            if done:
                break
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)

        return max_frame, total_reward, done, combined_info

    def reset(self):
        """Clear past frame buffer and init. to first obs. from inner env."""
        self._obs_buffer.clear()
        obs = self.env.reset()
        self._obs_buffer.append(obs)
        return obs

class ClipRewardEnv(MyWrapper):
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        reward = np.sign(reward)
        return obs, reward, done, info

class IgnoreNegativeRewardEnv(MyWrapper):
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        reward = max(reward, 0)
        return obs, reward, done, info

class ScaledRewardEnv(MyWrapper):
    def __init__(self, env, scale=1):
        MyWrapper.__init__(self, env)
        self.scale = scale

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        reward = reward*self.scale
        return obs, reward, done, info

class EpsGreedyEnv(MyWrapper):
    def __init__(self, env, eps=0.01):
        MyWrapper.__init__(self, env)
        self.eps = eps

    def step(self, action):
        if np.random.uniform()<self.eps:
            action = np.random.randint(self.env.action_space.n)
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, info

class StickyActionEnv(MyWrapper):
    def __init__(self, env, p=0.25):
        MyWrapper.__init__(self, env)
        self.p = p
        self.last_action = 0

    def step(self, action):
        if np.random.uniform() < self.p:
            action = self.last_action
        self.last_action = action
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, info

class Box(gym.Space):
    """
    A box in R^n.
    I.e., each coordinate is bounded.
    Example usage:
    self.action_space = spaces.Box(low=-10, high=10, shape=(1,))
    """
    def __init__(self, low, high, shape=None, dtype=np.uint8):
        """
        Two kinds of valid input:
            Box(-1.0, 1.0, (3,4)) # low and high are scalars, and shape is provided
            Box(np.array([-1.0,-2.0]), np.array([2.0,4.0])) # low and high are arrays of the same shape
        """
        if shape is None:
            assert low.shape == high.shape
            self.low = low
            self.high = high
        else:
            assert np.isscalar(low) and np.isscalar(high)
            self.low = low + np.zeros(shape)
            self.high = high + np.zeros(shape)
        self.dtype = dtype
    def contains(self, x):
        return x.shape == self.shape and (x >= self.low).all() and (x <= self.high).all()
    def to_jsonable(self, sample_n):
        return np.array(sample_n).tolist()
    def from_jsonable(self, sample_n):
        return [np.asarray(sample) for sample in sample_n]
    @property
    def shape(self):
        return self.low.shape
    @property
    def size(self):
        return self.low.shape
    def __repr__(self):
        return "Box" + str(self.shape)
    def __eq__(self, other):
        return np.allclose(self.low, other.low) and np.allclose(self.high, other.high)

class WarpFrame(MyWrapper):
    def __init__(self, env):
        """Warp frames to 84x84 as done in the Nature paper and later work."""
        MyWrapper.__init__(self, env)
        self.res = 84
        self.observation_space = Box(low=0, high=255, shape=(self.res, self.res, 1), dtype = np.uint8)

    def reshape_obs(self, obs):
        obs = np.dot(obs.astype('float32'), np.array([0.299, 0.587, 0.114], 'float32'))
        # print(obs.shape)
        obs = np.array(Image.fromarray(obs).resize((self.res, self.res),
                                                   resample=Image.BILINEAR), dtype=np.uint8)
        # print(obs.shape)
        return obs.reshape((self.res, self.res, 1))

    def reset(self):
        return self.reshape_obs(self.env.reset())

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self.reshape_obs(obs), reward, done, info

class MyResizeFrame(MyWrapper):
    def __init__(self, env):
        """Warp frames to 105x80"""
        MyWrapper.__init__(self, env)
        self.res = (105, 80, 3)
        self.net_res = (self.res[1], self.res[0], self.res[2])
        self.observation_space = Box(low=0, high=255, shape=self.net_res, dtype=np.uint8)

    def reshape_obs(self, obs):
        obs = np.array(Image.fromarray(obs).resize((self.res[0], self.res[1]),
                                                   resample=Image.BILINEAR), dtype=np.uint8)
        return obs

    def reset(self):
        return self.reshape_obs(self.env.reset())

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self.reshape_obs(obs), reward, done, info

class MyResizeFrameOld(MyWrapper):
    def __init__(self, env):
        """Warp frames to 105x80"""
        MyWrapper.__init__(self, env)
        self.res = (105, 80, 3)
        self.observation_space = Box(low=0, high=255, shape=self.res, dtype=np.uint8)

    def reshape_obs(self, obs):
        obs = np.array(Image.fromarray(obs).resize((self.res[0], self.res[1]),
                                                   resample=Image.BILINEAR), dtype=np.uint8)
        return obs.reshape(self.res)

    def reset(self):
        return self.reshape_obs(self.env.reset())

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self.reshape_obs(obs), reward, done, info

class FireResetEnv(MyWrapper):
    def __init__(self, env):
        """Take action on reset for environments that are fixed until firing."""
        MyWrapper.__init__(self, env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self):
        self.env.reset()
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset()
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset()
        return obs

class PreventSlugEnv(MyWrapper):
    def __init__(self, env, max_no_rewards=10000):
        """Abort if too much time without getting reward."""
        MyWrapper.__init__(self, env)
        self.last_reward = 0
        self.steps = 0
        self.max_no_rewards = max_no_rewards

    def step(self, *args, **kwargs):
        obs, reward, done, info = self.env.step(*args, **kwargs)
        self.steps += 1
        if reward > 0:
            self.last_reward = self.steps
        if self.steps - self.last_reward > self.max_no_rewards:
            done = True
        return obs, reward, done, info

    def reset(self):
        self.got_reward = False
        self.steps = 0
        return self.env.reset()

class VideoWriter(MyWrapper):
    def __init__(self, env, file_prefix):
        MyWrapper.__init__(self, env)
        self.file_prefix = file_prefix
        fd, self.temp_filename = tempfile.mkstemp('.mp4', 'tmp', '/'.join(self.file_prefix.split('/')[:-1]))
        os.close(fd)
        self.video_writer = None
        self.counter = 0
        self.orig_frames = None
        self.cur_step = 0
        self.is_train = hasattr(self.env, 'demo_replay_info')
        self.score = 0
        self.randval = 0

    def process_frame(self, frame):
        f_out = np.zeros((224, 160, 3), dtype=np.uint8)
        f_out[7:-7, :] = np.cast[np.uint8](frame)
        return f_out

    def get_orig_frame(self, shape):
        if self.orig_frames is None:
            self.orig_frames = {}
            tmp = self.env.env.unwrapped.clone_state()

            self.env.env.reset()
            for i, a in enumerate(self.env.actions):
                self.orig_frames[i + self.env.reset_steps_ignored] = self.env.env.render(mode='rgb_array')
                self.env.unwrapped.ale.act(a)

            self.orig_frames[len(self.env.actions)] = self.env.env.render(mode='rgb_array')

            self.env.env.unwrapped.restore_state(tmp)

        if self.cur_step not in self.orig_frames:
            frame = np.zeros(shape, dtype=np.uint8)
            frame[:, :, 1] = 255
            return frame

        return self.orig_frames[self.cur_step]

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.cur_step += 1

        if reward <= -999000:
            reward = 0
        self.score += reward

        self.video_writer.append_data(self.process_frame(obs))
        return obs, reward, done, info

    def reset(self):
        self.score = 0
        if self.video_writer is not None:
            self.video_writer.close()
            if self.is_train:
                demo_idx = getattr(self.env, 'cur_demo_idx', 0)
                filename = f'{self.file_prefix}_demo-{demo_idx}_start-{self.env.demo_replay_info[demo_idx].starting_point_current_ep if self.is_train else 0}_{random.randint(0, 2)}.mp4'
                os.rename(self.temp_filename, filename)
            else:
                n_noops = -1
                try:
                    n_noops = self.recursive_getattr('cur_noops')
                except Exception:
                    pass
                os.rename(self.temp_filename, self.file_prefix + str(n_noops) + '_'+ str(self.randval) + '.mp4')
            self.counter += 1
        res = self.env.reset()
        self.randval = random.randint(0, 1000000)
        self.video_writer = imageio.get_writer(self.temp_filename, mode='I', fps=120)
        if self.is_train:
            self.cur_step = self.env.demo_replay_info[self.env.cur_demo_idx].starting_point_current_ep
        return res

def my_wrapper(env,
               clip_rewards=True,
               frame_resize_wrapper=MyResizeFrame,
               scale_rewards=None,
               ignore_negative_rewards=False,
               sticky=True):
    assert 'NoFrameskip' in env.spec.id
    assert not (clip_rewards and scale_rewards), "Clipping and scaling rewards makes no sense"
    if scale_rewards is not None:
        env = ScaledRewardEnv(env, scale_rewards)
    if clip_rewards:
        env = ClipRewardEnv(env)
    if sticky:
        env = StickyActionEnv(env)
    if ignore_negative_rewards:
        env = IgnoreNegativeRewardEnv(env)
    env = MaxAndSkipEnv(env, skip=4)
    if 'Pong' in env.spec.id:
        env = FireResetEnv(env)
    env = frame_resize_wrapper(env)
    return env

def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]

def worker(remote, env_fn_wrapper):
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            ob, reward, done, info = env.step(data)
            if done:
                ob = env.reset()
            remote.send((ob, reward, done, info))
        elif cmd == 'reset':
            ob = env.reset()
            remote.send(ob)
        elif cmd == 'close':
            remote.close()
            break
        elif cmd == 'get_spaces':
            remote.send((env.action_space, env.observation_space))
        elif cmd == 'get_history':
            senv = env
            while not hasattr(senv, 'get_history'):
                senv = senv.env
            remote.send(senv.get_history(data))
        elif cmd == 'recursive_getattr':
            remote.send(env.recursive_getattr(data))
        elif cmd == 'decrement_starting_point':
            env.decrement_starting_point(*data)
        else:
            raise NotImplementedError

class CloudpickleWrapper(object):
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """
    def __init__(self, x):
        self.x = x
    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)
    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)

class SubprocVecEnv(MyWrapper):
    def __init__(self, env_fns):
        """
        envs: list of gym environments to run in subprocesses
        """
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [Process(target=worker, args=(work_remote, CloudpickleWrapper(env_fn)))
            for (work_remote, env_fn) in zip(self.work_remotes, env_fns)]
        for p in self.ps:
            p.start()

        self.remotes[0].send(('get_spaces', None))
        self.action_space, self.observation_space = self.remotes[0].recv()

    def step(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        results = [remote.recv() for remote in self.remotes]
        obs, rews, dones, infos = zip(*results)

        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def reset_task(self):
        for remote in self.remotes:
            remote.send(('reset_task', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def get_history(self, nsteps):
        for remote in self.remotes:
            remote.send(('get_history', nsteps))
        results = [remote.recv() for remote in self.remotes]
        obs, acts, dones = zip(*results)
        obs = np.stack(obs)
        acts = np.stack(acts)
        dones = np.stack(dones)
        return obs, acts, dones

    def recursive_getattr(self, name):
        for remote in self.remotes:
            remote.send(('recursive_getattr',name))
        return [remote.recv() for remote in self.remotes]

    def decrement_starting_point(self, n, idx):
        for remote in self.remotes:
            remote.send(('decrement_starting_point', (n, idx)))

    def close(self):
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()

    @property
    def num_envs(self):
        return len(self.remotes)

class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, cur_noops=0, n_envs=1, save_path=None):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        gym.Wrapper.__init__(self, env)
        self.override_num_noops = None
        self.noop_action = 0
        self.cur_noops = cur_noops - n_envs
        self.n_envs = n_envs
        self.save_path = save_path
        self.score = 0
        self.levels = 0
        self.in_treasure = False
        self.rewards = []
        self.actions = []
        self.rng = np.random.RandomState(os.getpid())
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def choose_noops(self):
        n_done = []
        for i in range(0, 31):
            json_path = self.save_path + '/' + str(i) + '.json'
            n_data = 0
            try:
                import json
                n_data = len(json.load(open(json_path)))
            except Exception:
                pass
            n_done.append(n_data)

        weights = np.array([(max(0.00001, 5 - e)) for e in n_done])
        div = np.sum(weights)
        if div == 0:
            weights = np.array([1 for e in n_done])
            div = np.sum(weights)
        return np.random.choice(list(range(len(n_done))), p=weights/div)

    def reset(self, **kwargs):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        noops = self.choose_noops()
        obs = self.env.reset(**kwargs)
        assert noops >= 0
        self.cur_noops = noops
        self.env.cur_noops = noops
        self.score = 0
        self.rewards = []
        self.actions = []
        self.in_treasure = False
        self.levels = 0
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        a, reward, done, c = self.env.step(ac)
        from collections import Counter
        in_treasure = Counter(a[:, :, 2].flatten()).get(136, 0) > 20_000
        if self.in_treasure and not in_treasure:
            self.levels += 1
        self.in_treasure = in_treasure
        if reward <= -999000:
            reward = 0
        self.actions.append(ac)
        self.rewards.append(reward)
        self.score += reward

        if self.save_path and done:
            json_path = self.save_path + '/' + str(self.cur_noops) + '.json'
            if 'episode' not in c:
                c['episode'] = {}
            if 'write_to_json' not in c['episode']:
                c['episode']['write_to_json'] = []
                c['episode']['json_path'] = json_path
            c['episode']['l'] = len(self.actions)
            c['episode']['r'] = self.score
            c['episode']['as_good_as_demo'] = False
            c['episode']['starting_point'] = 0
            c['episode']['idx'] = 0
            c['episode']['write_to_json'].append({'score': self.score, 'levels': self.levels, 'actions': [int(e) for e in self.actions]})
        return a, reward, done, c






def wrap(env):
    """Apply a common set of wrappers for Atari games."""
    assert 'NoFrameskip' in env.spec.id
    # env = EpisodicLifeEnv(env)
    env = NoopResetEnv(env, noop_max=30)
    fs = 4
    if "SpaceInvaders" in env.spec.id:
        fs = 3
    env = MaxAndSkipEnv(env, skip=fs)
    env = ProcessFrame84(env)
    env = FrameStack(env, 4)
    # env = ClippedRewardsWrapper(env)
    return env

def make_env(env_id,env_num,ARGS):
    env_list = []
    for rank in range(env_num):
        env = gym.make(env_id)
        env = wrap(env_id)
        env.seed(ARGS.seed + rank)
        env_list.append(env)
    return env_list

# here put the import lib
import torch
import numpy as np

from torchvision import transforms
from collections import deque


trans = transforms.Compose(
    [transforms.Grayscale(), transforms.Resize((84, 84)), transforms.ToTensor()]
)


class ProcessUnit(object):
    """
    Preprocessing procedures to transfrom Atari raw screen pixel to input of network   
        1. Take themaximum value for each pixel colour   
        2. Grayscale and resize 84x84   
        3. Turn 4(or 3) frame into 1 tensor 84x84x4   
    Ref: Mnih, Volodymyr, et al. "Human-level control through deep reinforcement learning." Nature 518.7540 (2015): 529-533.\n
    Url: https://www.nature.com/articles/nature14236   
    
    Attributes:\n
        self.length(int):       Length of queue (length * frame_skip)
        self.frame_list(list):  Queue of frame
    """

    def __init__(self, length=4, frame_skip=4):
        self.length = length * frame_skip
        self.frame_list = deque(maxlen=self.length)

    def step(self, x):
        """add a frame to frame_list"""
        # insert in left
        self.frame_list.appendleft(x)

    def to_torch_tensor(self):
        """Turn 4 frames into 1 tensor (84x84x4)"""
        length = len(self.frame_list)
        x_list = []
        i = 0
        while i < length:
            if i == length - 1:
                x_list.append(self.transform(self.frame_list[i]))
            else:
                x = np.maximum(self.frame_list[i], self.frame_list[i + 1])
                x_list.append(self.transform(x))
            i += 4
        while len(x_list) < 4:
            x_list.append(x_list[-1])
        return torch.cat(x_list, 1)

    def transform(self, x):
        """pytorch transform"""
        x = transforms.ToPILImage()(x).convert("RGB")
        x = trans(x)
        x = x.reshape(1, 1, 84, 84)
        return x
