
# Modifications Copyright (c) 2019 Uber Technologies, Inc.

import argparse
import os
import numpy as np
import gym

def train(game_name,
          policy,
          num_timesteps,
          lr,
          entropy_coef,
          load_path,
          starting_point,
          save_path,
          allowed_lag,
          gamma,
          move_threshold,
          demo,
          start_frame,
          game_over_on_life_loss,
          allowed_score_deficit,
          frame_resize="MyResizeFrame",
          steps_per_demo=1024,
          clip_rewards=True,
          scale_rewards=None,
          ignore_negative_rewards=False,
          sticky=False,
          test_from_start=False):
    import tensorflow as tf
    import horovod.tensorflow as hvd
    hvd.init()
    print('initialized worker %d' % hvd.rank(), flush=True)
    from baselines.common import set_global_seeds
    set_global_seeds(hvd.rank())
    from atari_reset.ppo import learn
    from atari_reset.policies import CnnPolicy, GRUPolicy
    from atari_reset.wrappers import ReplayResetEnv, ResetManager, SubprocVecEnv, VideoWriter, VecFrameStack, my_wrapper, MyResizeFrame, WarpFrame, MyResizeFrameOld

    if frame_resize == "MyResizeFrame":
        frame_resize_wrapper = MyResizeFrame
    elif frame_resize == "WarpFrame":
        frame_resize_wrapper = WarpFrame
    elif frame_resize == "MyResizeFrameOld":
        frame_resize_wrapper = MyResizeFrameOld
    else:
        raise NotImplementedError("No such frame-size wrapper: " + frame_resize)
    ncpu = 2
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=ncpu,
                            inter_op_parallelism_threads=ncpu)
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = str(hvd.local_rank())
    tf.Session(config=config).__enter__()

    nrstartsteps = 320  # number of non frameskipped steps to divide workers over
    nenvs = 16
    nrworkers = hvd.size() * nenvs
    workers_per_sp = int(np.ceil(nrworkers / nrstartsteps))

    # load demo
    if demo is None:
        demo = 'demos/'+game_name+'.demo'
    print('Using demo', demo)
    def make_env(rank):
        def env_fn():
            env = gym.make(game_name + 'NoFrameskip-v4')
            env = ReplayResetEnv(env,
                                 demo_file_name=demo,
                                 seed=rank,
                                 reset_steps_ignored=start_frame,
                                 workers_per_sp=workers_per_sp,
                                 game_over_on_life_loss=game_over_on_life_loss,
                                 allowed_lag=allowed_lag,
                                 allowed_score_deficit=allowed_score_deficit,
                                 test_from_start=test_from_start)
            if rank%nenvs == 0 and hvd.local_rank() == 0: # write videos during training to track progress
                dir = os.path.join(save_path, game_name)
                os.makedirs(dir, exist_ok=True)
                videofile_prefix = os.path.join(dir, 'episode')
                env = VideoWriter(env, videofile_prefix)
            env = my_wrapper(env,
                             clip_rewards=clip_rewards,
                             frame_resize_wrapper=frame_resize_wrapper,
                             scale_rewards=scale_rewards,
                             ignore_negative_rewards=ignore_negative_rewards,
                             sticky=sticky)
            return env
        return env_fn

    env = SubprocVecEnv([make_env(i + nenvs * hvd.rank()) for i in range(nenvs)])
    env = ResetManager(env, move_threshold=move_threshold, steps_per_demo=steps_per_demo)
    if starting_point is not None:
        env.set_max_starting_point(starting_point, 0)
    env = VecFrameStack(env, 4)

    # load policy
    policy = {'cnn' : CnnPolicy, 'gru': GRUPolicy}[policy]
    learn(policy=policy, env=env, nsteps=128, lam=.95, gamma=gamma, noptepochs=4, log_interval=1, save_interval=100,
          ent_coef=entropy_coef, l2_coef=1e-7, lr=lr, cliprange=0.1, total_timesteps=num_timesteps,
          norm_adv=True, load_path=load_path, save_path=save_path, game_name=game_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--game', type=str, default='MontezumaRevenge')
    parser.add_argument('--num_timesteps', type=int, default=1e12)
    parser.add_argument('--policy', default='gru')
    parser.add_argument('--load_path', type=str, default=None, help='Path to load existing model from')
    parser.add_argument('--starting_point', type=int, default=None,
                        help='Demo-step to start training from, if not the last')
    parser.add_argument('--save_path', type=str, default='results', help='Where to save results to')
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--entropy_coef', type=float, default=1e-4)
    parser.add_argument('--allowed_lag', type=int, default=50)
    parser.add_argument('--gamma', type=float, default=0.999)
    parser.add_argument('--move_threshold', type=float, default=0.2)
    parser.add_argument('--demo', type=str, default=None)
    parser.add_argument('--start_frame', type=int, default=0,
                        help='Training will start this many frames back')
    parser.add_argument('--no_game_over_on_life_loss', action='store_true', default=False,
                        help='Whether the agent is allowed to continue after a life loss.')
    parser.add_argument('--allowed_score_deficit', type=int, default=0,
                        help='The imitator is allowed to be this many points worse than the example')
    parser.add_argument('--frame_resize', type=str, default="MyResizeFrame",
                        help='Resize wrapper to be used for the game.')
    parser.add_argument('--steps_per_demo', type=int, default=1024)
    parser.add_argument('--no_clip_rewards', action='store_true', default=False)
    parser.add_argument('--scale_rewards', type=float, default=None)
    parser.add_argument('--ignore_negative_rewards', action='store_true', default=False)
    parser.add_argument("--sticky", help="Use sticky actions", action="store_true")
    parser.add_argument("--test_from_start", action="store_true", default=False,
                        help="Add a virtual demo that always runs from the start")
    args = parser.parse_args()

    train(args.game,
          args.policy,
          args.num_timesteps,
          args.learning_rate,
          args.entropy_coef,
          args.load_path,
          args.starting_point,
          args.save_path,
          args.allowed_lag,
          args.gamma,
          args.move_threshold,
          args.demo,
          args.start_frame,
          not args.no_game_over_on_life_loss,
          args.allowed_score_deficit,
          args.frame_resize,
          args.steps_per_demo,
          not args.no_clip_rewards,
          args.scale_rewards,
          args.ignore_negative_rewards,
          args.sticky)
