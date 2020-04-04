
import torch.multiprocessing as mp
from src.env import make_env
from src.agent import PPO
from src.loaddemo import Demo, ResetDemo

class ARGS:
    gamename = 'AlienDeterministic'
    M = 10 # M parallel worker
    delta = 10 # delta start point shift size 
    rho = 0.1 # success threhold rho
    T = 1024 # demostration length T
    algorithm = "PPO" # laerning algorithm
    
    ncpu = 1
    # fixed
    seed = 111

    @staticmethod
    def set_params(kwargs):
        ARGS.gamename = kwargs['game']+'Deterministic'
        ARGS.M = kwargs['worker_num']
        ARGS.delta = kwargs['shift_size']
        ARGS.rho = kwargs['success_threhold']
        ARGS.T = kwargs['demo_len']
        ARGS.ncpu = kwargs['ncpu']

def main(kwargs):

    # TODO:Initilization:
    # 1. multiprocess setting
    pool = mp.Pool(ARGS.ncpu)
    # 2. env setting
    env_list = make_env(ARGS.gamename,ARGS.worker_num)
    # 3. load demo
    demo = Demo("./demos/Pitfall.demo")
    # 4. model setting
    # 5. runner setting
    # 6. gpu config
    ARGS = ARGS()
    ARGS.set_params(kwargs)
    workers = [PPO(env_list[rank],ARGS.seed+rank,ARGS) for rank in range(ARGS.M)]

    
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
        actor_model,critic_model = optimize(actor_model,critic_model,D)
        for w in workers:
            w.update(actor_model,critic_model,reset_point)

        # TODO:output log info(rank == 0)

