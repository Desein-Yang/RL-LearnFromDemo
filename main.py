import logging
import os
import torch.multiprocessing as mp
from src.env import make_atari_env,ReplayResetEnv
from src.agent import Worker,PPO_Optimizer
from src.loaddemo import Demo,Buffer
from src.utils import mk_folder,setup_logging

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

def main(kwargs,ARGS):

    # TODO:Initilization:
    # 1. multiprocess setting
    pool = mp.Pool(ARGS.ncpu)

    # 2. env setting
    env_list = make_atari_env(ARGS.gamename,ARGS.M,ARGS.seed)
    
    # 3. load demo
    demo = Demo("./demos/Pitfall.demo")
    
    # 4. model setting
    # 5. runner setting
    # 6. gpu config
    # 7. args setting
    ARGS = ARGS()
    ARGS.set_params(kwargs)
    folderpath = mk_folder(os.getcwd(),ARGS)
    ARGS.set_folder_path(folderpath)
    print("start!")
    print("-----------------------------------------")

    logger = logging.getLogger(__name__)
    filename = ARGS.kwargs.game+'-'+'M'+'-'+ARGS.M+'-'+'delta'+'-'+ARGS.delta+'-'+'rho'+ARGS.rho+'.txt'
    logger = setup_logging(logger,folderpath,filename)
    ARGS.set_logger(logger)
    logger.info("Log initialized")

    workers = [Worker(env_list[rank],ARGS) for rank in range(ARGS.M)]
    optimizer = PPO_Optimizer(ARGS)
    logger.info("Generate %d worker and optimizer" % ARGS.M)

    D = []
    W = []
    reset_point = ARGS.T
    while (reset_point > 0):
        # TODO:run main loop(parallel worker)
        # gather data start state for worker T* and reset to st*
        # for step in 1,..L-1:
        #     sample action and receive reward/copy from demonstration
        #     put {action,reward,state,done,info} in buffer
        #     if done:
        #         if better than demonstration:
        #               W+=1
        #     sample start state T* and reset to st*
        # send D and W to optimizer
        jobs = [
            pool.apply_async(w.rollout,(demo,ARGS)) 
            for w in workers
            ]
        
        for j in jobs:
            D.append(j.gets[0])
            W.append(j.gets[1])

        # TODO:optimize
        # if sum(W)/sum(D)>p:
        #     T shift forward for delta(shift size)
        # optimize params by PPO or Other
        # TODO: sum(demo) means didt length, it is not clear
        if sum(W) / sum(demo) >= ARGS.rho:
            reset_point = reset_point - ARGS.delta
        
        w = workers[0]
        actor_model,critic_model = optimizer.optimize(w.actor_model,w.critic_model,D)
        for w in workers:
            w.update(actor_model,critic_model,reset_point)
            
            
        # TODO:output log info(rank == 0)

def args_parser():
    import argparse
    parser=argparse.ArgumentParser(description='Please input enviroment')
    
    parser.add_argument('--game',nargs='?',default='CartPole',help='select the environment')
    parser.add_argument('--ncpu',nargs='?',default= 1,help='select the hardware')
    parser.add_argument('--worker_num',nargs='?',default=10,help='')
    parser.add_argument('--success_threhold',nargs='?',default=10,help='')
    parser.add_argument('--demo_len',nargs='?',default=10,help='')    
    parser.add_argument('--shift_size',nargs='?',default=10,help='')
    
    args=parser.parse_args()
    return args

if __name__ == "__main__":
    args = args_parser()
    main(args,ARGS)
    print("finish : %s for game:%s" % (str(args), args.game))
        

