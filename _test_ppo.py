from cfg import get_cfg
import os
import pandas as pd
#from dqn import *
import random

import vessl
from torch.utils.tensorboard import SummaryWriter
from agent.ppo import *
from environment.env import UPMSP

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

if __name__ == "__main__":
    writer = SummaryWriter()
    vessl.init()

    cfg = get_cfg()
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    num_episode = cfg.n_episode
    episode = 1
    score_avg = 0
    state_size =104

    mode = cfg.mode
    if mode == 'heuristic':
        action_size = 3
    else:
        action_size = 12

    lr = cfg.lr
    gamma = cfg.gamma
    lmbda = cfg.lmbda
    n_job = cfg.n_job
    eps_clip = cfg.eps_clip
    K_epoch = cfg.K_epoch
    T_horizon = cfg.T_horizon

    w_delay = cfg.w_delay
    w_move = cfg.w_move
    w_priority = cfg.w_priority
    state_size = 104        # feature 1~8 size = 104 / 176
    log_path = '../result/model/dqn'
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    event_path = '../environment/result/dqn'
    if not os.path.exists(event_path):
        os.makedirs(event_path)

    #env = UPMSP(log_dir=event_path, num_j=1500,num_m=8, action_number = action_size, action_mode = 'WCOVERT')
    env = UPMSP(log_dir=event_path, num_j=n_job, num_m=8, action_number=action_size, min=0.1, max=4, action_mode=mode)  # action_mode 바꿔야 함 heuristic, WCOVERT
    agent = Agent(state_size, action_size, lr, gamma, lmbda, eps_clip, K_epoch)
    #agent.network.load_state_dict(torch.load('../result/model/dqn/episode8000.pt')["model_state_dict"])    # WCOVERT PPO
    # agent.network.load_state_dict(torch.load('../result/model/ppo/episode6200.pt')["model_state_dict"])   # heuristic PPO
    # agent.network.load_state_dict(torch.load('../result/model/ppo/episode4800_atc_ppo12.pt')["model_state_dict"])     # ATC PPO
    #agent.network.load_state_dict(torch.load('../result/model/ppo/episode200_heuristic_ppo_3.pt')["model_state_dict"])     # heuristic PPO



    # q = Qnet(state_size, action_size)
    # q.load_state_dict(torch.load(log_path + '/episode5000.pt')["model_state_dict"])       # 가중치가 저장되는 경로

    tard_list = list()
    np.random.seed(42)
    random.seed(42)
    #print(num_episode)
    with open("/output/output.txt", "w") as file:
        for i in range(num_episode):
            env.e = i
            step = 0
            done = False
            s = env.reset()
            r = list()
            possible_actions = [True] * action_size
            while not done:
                epsilon = 0
                step += 1
                #a = 2   # 0: "WSPT", 1: "WMDD", 2: "ATC", 3: "WCOVERT"
                a, prob, mask = agent.get_action(s, possible_actions)
    
                # 환경과 연결
                next_state, reward, done = env.step(a)
    
                r.append(reward)
                s = next_state
    
                if done:
                    env.monitor.save_tracer()
                    break
    
            mean_wt = env.monitor.tardiness / env.num_job
            tard_list.append(mean_wt)
            print("{} {}".format(i+1, -mean_wt))
            vessl.log(step=i+1, payload={'MWT': np.mean(-mean_wt)})
            file.write(str(-mean_wt)+'\n')
            # print("Episode {0} | MWT = {1} | CUM_MWT = {2}".format(i+1, -mean_wt, -np.mean(tard_list)))
    
        print("Total Mean Weighted Tardiness = ", np.mean(tard_list))
