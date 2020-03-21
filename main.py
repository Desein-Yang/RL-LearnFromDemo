
from env import make_env
from agent import PPO

class ARGS:
    gamename = 'AlienDeterministic'
    worker_num = 10 # M parallel worker

def main():

    # TODO:Initilization:
    args = ARGS()
    # 1. multiprocess setting
    # 2. env setting
    env_list = make_env(args.gamename,args.worker_num)
    # 3. load demo
    # 4. model setting
    # 5. runner setting
    # 6. gpu config
    workers = [PPO(env_list[rank],args.seed+rank) for rank in range(args.worker_num)]

    # TODO:run main loop(parallel worker)
    # sample start state for worker T* and reset to st*
    # for step in 1,..L-1:
    #     sample action and receive reward/copy from demonstration
    #     put {action,reward,state,done,info} in buffer
    #     if done:
    #         if better than demonstration:
    #               W+=1
    #     sample start state T* and reset to st*
    # send D and W to optimizer
    for 


    # TODO:optimize
    # if sum(W)/sum(D)>p:
    #     T shift forward for delta(shift size)
    # optimize params by PPO or Other


    # TODO:output log info(rank == 0)
