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
                self.returns = np.cumsum(self.rewards)
                assert len(self.rewards) == len(self.actions)
                self.lives = dat['lives']
                self.obs = dat['checkpoints']
                self.checkpoint_action_nr = dat['checkpoint_action_nr']
        self.length = len(self.actions)

    def replay(self,i):
        """return action, ob, reward, done, info in demo.  
        info??"""
        if self.lives[i] == 0:
            done = True
        else:
            done = False
        return self.actions[i],self.obs[i],self.rewards[i],done,info
                                              
class ResetDemo:
    """reset env into start point"""
    def __init__(self,env,i):
        self.env = env
        self.idx = i

    
    
# ------------------------ origin --------------------
# class ResetDemoInfo:
#     def __init__(self, env, idx):
#         self.env = env
#         self.idx = idx
#         starting_points = self.env.recursive_getattr(f'starting_point_{idx}')
#         all_starting_points = flatten_lists(MPI.COMM_WORLD.allgather(starting_points))
#         # all_starting_points = 
#         self.min_starting_point = min(all_starting_points)
#         self.max_starting_point = max(all_starting_points)
#         self.nrstartsteps = self.max_starting_point - self.min_starting_point
#         assert(self.nrstartsteps > 10)
#         self.max_max_starting_point = self.max_starting_point
#         self.starting_point_success = np.zeros(self.max_starting_point+10000)
#         self.infos = []

# class DemoReplay:
#     def __init__(self,demo_file_name):
#         assert demo_file_name is not None
#         with open(demo_file_name, "rb") as f:
#             dat = pickle.load(f)
#         self.actions = dat['actions']
#         self.rewards = dat['rewards']
#         self.returns = np.cumsum(self.rewards)
#         assert len(self.rewards) == len(self.actions)
#         self.lives = dat['lives']

#         self.checkpoints = dat['checkpoints']
#         self.checkpoint_action_nr = dat['checkpoint_action_nr']
#         self.starting_point = len(self.actions) - 1 - seed//workers_per_sp
#         self.starting_point_current_ep = None  

#         #TODO:what are seed and workers per sp          

# class DemoReplayInfo:
#     def __init__(self, demo_file_name, seed, workers_per_sp):
#         # Added to allow for the creation of "fake" replay information
#         if demo_file_name is None:
#             self.actions = None
#             self.returns = [0]
#             self.checkpoints = None
#             self.checkpoint_action_nr = None
#             self.starting_point = 0
#             self.starting_point_current_ep = None
#         else:
#             with open(demo_file_name, "rb") as f:
#                 dat = pickle.load(f)
#             self.actions = dat['actions']
#             rewards = dat['rewards']
#             assert len(rewards) == len(self.actions)
#             self.returns = np.cumsum(rewards)
#             self.checkpoints = dat['checkpoints']
#             self.checkpoint_action_nr = dat['checkpoint_action_nr']
#             self.starting_point = len(self.actions) - 1 - seed//workers_per_sp
#             self.starting_point_current_ep = None

# class ReplayResetEnv():
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
#         self.actions_to_overwrite = []
#         self.frac_sample = frac_sample
#         self.game_over_on_life_loss = game_over_on_life_loss
#         self.allowed_lag = allowed_lag
#         self.allowed_score_deficit = allowed_score_deficit
#         self.demo_replay_info = []
#         self.test_from_start = test_from_start
#         if test_from_start:
#             self.demo_replay_info.append(DemoReplayInfo(None, seed, workers_per_sp))
#         if os.path.isdir(demo_file_name):
#             import glob
#             for f in sorted(glob.glob(demo_file_name + '/*.demo')):
#                 self.demo_replay_info.append(DemoReplayInfo(f, seed, workers_per_sp))
#         else:
#             self.demo_replay_info.append(DemoReplayInfo(demo_file_name, seed, workers_per_sp))
#         self.cur_demo_replay = None
#         self.cur_demo_idx = -1
#         self.extra_frames_counter = -1
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



# class ResetManager(MyWrapper):
#     def __init__(self, env, ARGS):
#         super(ResetManager, self).__init__(env)    
#         self.n_demos = ARGS.M
#         self.demos = [ResetDemoInfo(self.env,idx) for idx in range(self.n_demos)]
#         self.delta = ARGS.delta
#         self.demo_len = ARGS.T
    
#     def move_start_point(self,):
#         pass

# class ResetManager1(MyWrapper):
    #def __init__(self, env, move_threshold=0.2, steps_per_demo=1024):
    #     super(ResetManager, self).__init__(env)
    #     self.n_demos = self.recursive_getattr('n_demos')[0]
    #     self.demos = [ResetDemoInfo(self.env, idx) for idx in range(self.n_demos)]
    #     self.counter = 0
    #     self.move_threshold = move_threshold
    #     self.steps_per_demo = steps_per_demo

    # def proc_infos(self):
    #     for idx in range(self.n_demos):
    #         epinfos = [info['episode'] for info in self.demos[idx].infos if 'episode' in info]

    #         if hvd.size()>1:
    #             epinfos = flatten_lists(MPI.COMM_WORLD.allgather(epinfos))

    #         new_sp_wins = {}
    #         new_sp_counts = {}
    #         for epinfo in epinfos:
    #             sp = epinfo['starting_point']
    #             if sp in new_sp_counts:
    #                 new_sp_counts[sp] += 1
    #                 if epinfo['as_good_as_demo']:
    #                     new_sp_wins[sp] += 1
    #             else:
    #                 new_sp_counts[sp] = 1
    #                 if epinfo['as_good_as_demo']:
    #                     new_sp_wins[sp] = 1
    #                 else:
    #                     new_sp_wins[sp] = 0

    #         for sp,wins in new_sp_wins.items():
    #             self.demos[idx].starting_point_success[sp] = np.cast[np.float32](wins)/new_sp_counts[sp]

    #         # move starting point, ensuring at least 20% of workers are able to complete the demo
    #         csd = np.argwhere(np.cumsum(self.demos[idx].starting_point_success) / self.demos[idx].nrstartsteps >= self.move_threshold)
    #         if len(csd) > 0:
    #             new_max_start = csd[0][0]
    #         else:
    #             new_max_start = np.minimum(self.demos[idx].max_starting_point + 100, self.demos[idx].max_max_starting_point)
    #         n_points_to_shift = self.demos[idx].max_starting_point - new_max_start
    #         self.decrement_starting_point(n_points_to_shift, idx)
    #         self.demos[idx].infos = []

    # def decrement_starting_point(self, n_points_to_shift, idx):
    #     self.env.decrement_starting_point(n_points_to_shift, idx)
    #     starting_points = self.env.recursive_getattr(f'starting_point_{idx}')
    #     all_starting_points = flatten_lists(MPI.COMM_WORLD.allgather(starting_points))
    #     self.demos[idx].max_starting_point = max(all_starting_points)

    # def set_max_starting_point(self, starting_point, idx):
    #     n_points_to_shift = self.demos[idx].max_starting_point - starting_point
    #     self.decrement_starting_point(n_points_to_shift, idx)

    # def step(self, action):
    #     obs, rews, news, infos = self.env.step(action)
    #     for info in infos:
    #         self.demos[info['idx']].infos.append(info)
    #     self.counter += 1
    #     if self.counter > (self.demos[0].max_max_starting_point - self.demos[0].max_starting_point) / 2 and self.counter % (self.steps_per_demo * self.n_demos) == 0:
    #         self.proc_infos()
    #     return obs, rews, news, infos

    # def step_wait(self):
    #     obs, rews, news, infos = self.env.step_wait()
    #     for info in infos:
    #         self.demos[info['idx']].infos.append(info)
    #     self.counter += 1
    #     if self.counter > (self.demos[0].max_max_starting_point - self.demos[0].max_starting_point) / 2 and self.counter % (self.steps_per_demo * self.n_demos) == 0:
    #         self.proc_infos()
    #     return obs, rews, news, infos

if __name__ == "__main__":
    demo = Demo('Phase2\demos\Pitfall.demo')
    print(demo)
