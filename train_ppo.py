import os
import vessl
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter
from environment.env import UPMSP
from cfg import get_cfg
from agent.ppo import *
import random



if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    #writer = SummaryWriter()
    cfg = get_cfg()
    vessl.init(organization="snu-eng-dgx", project="Quay", hp=cfg)

    lr = cfg.lr
    gamma = cfg.gamma
    lmbda = cfg.lmbda
    eps_clip = cfg.eps_clip
    K_epoch = cfg.K_epoch
    T_horizon = cfg.T_horizon

    w_delay = cfg.w_delay
    w_move = cfg.w_move
    w_priority = cfg.w_priority
    state_size = 104        # feature 1~8 size = 104 / 176
    action_size = 4
    # rewards_list = list()



    model_dir = '/output/'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    log_path = 'result/model/ppo'
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    event_path = 'environment/result/ppo'
    if not os.path.exists(event_path):
        os.makedirs(event_path)
    range_min = 0.1
    range_max = 4
    env = UPMSP(log_dir=event_path, num_j=1000, num_m=8, action_number=action_size, action_mode='heuristic', min = range_min, max = range_max)
    agent = Agent(state_size, action_size, lr, gamma, lmbda, eps_clip, K_epoch)


    # agent.network.load_state_dict(torch.load('output/train/model/episode6400.pt')["model_state_dict"])
    # agent.optimizer.load_state_dict(torch.load('output/train/model/episode6400.pt')["optimizer_state_dict"])
    # k = 6950
    k = 1

    # writer = SummaryWriter(log_dir)
    possible_actions = [True] * action_size

    for e in range(k, cfg.n_episode + 1):


        s = env.reset()
        update_step = 0
        r_epi = 0.0
        avg_loss = 0.0
        done = False

        while not done:
            for t in range(T_horizon):
                #possible_actions = env.get_possible_actions()
                a, prob, mask = agent.get_action(s, possible_actions)
                s_prime, r, done = env.step(a)
                agent.put_data((s, a, r, s_prime, prob[a].item(), mask, done))
                r_epi += r
                if done:
                    break
            update_step += 1
            avg_loss += agent.train()
        agent.scheduler.step()
        vessl.log(step=e, payload={'reward': np.mean(r_epi)})
        
        # rewards_list.append(r_epi)
        # moving_average_duration = 80
        # if e % 100 == 0:
        #     if len(rewards_list) >= moving_average_duration:
        #         moving_average = [np.mean(rewards_list[i:i+moving_average_duration]) for i in range(0, len(rewards_list)-moving_average_duration)]
        #         r_df = pd.DataFrame(rewards_list)
        #         me_df = pd.DataFrame(moving_average)
        #         r_df.to_csv('r_df_min_{}_max_{}_action_size_{}.csv'.format(range_min, range_max, action_size))
        #         me_df.to_csv('me_df_min_{}_max_{}_action_size_{}.csv'.format(range_min, range_max, action_size))
        print(e, r_epi)
        if e % 100 == 0:
            agent.save_network(e, model_dir)
