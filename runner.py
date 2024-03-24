from tqdm import tqdm
from agent import Agent
from common.replay_buffer import Buffer
import torch
import os
import numpy as np
import os
import seaborn as sns
import pandas as pd

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d import Axes3D
import os
from PIL import Image
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

class Runner:
    def __init__(self, args, env):
        self.args = args
        args.noise_rate = 0.05
        args.epsilon = 0.1
        args.max_episode_len = 120
        args.time_steps = 1000000

        self.noise = args.noise_rate
        self.epsilon = args.epsilon
        self.episode_limit = args.max_episode_len
        self.env = env
        self.agents = self._init_agents()
        print('BinGo')
        self.buffer = Buffer(args)
        self.save_path = self.args.save_dir + '/' + self.args.scenario_name
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def _init_agents(self):
        agents = []
        for i in range(self.args.n_agents):
            #print('THE OUTPUT:',agents)
            agent = Agent(i, self.args)
            agents.append(agent)
        
        return agents

    def run(self):
        returns = []
        bufferINFO = []
        done = False
        data_path = './data/flocking data/' + str(self.args.n_agents) + 'agents'
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        reward_path = './data/reward data/' + str(self.args.n_agents) + 'agents'
        if not os.path.exists(reward_path):
            os.makedirs(reward_path)
        for time_step in tqdm(range(self.args.time_steps)):
            # reset the environment
            if (time_step % self.episode_limit == 0) :
                s = self.env.reset()
                if (time_step!=0):
                    bufferINFO=np.array(bufferINFO)

                    np.save(data_path + '/' + 'flocking' + str(time_step), bufferINFO)
                    x=np.ones((len(bufferINFO),len(bufferINFO[0])))
                    y=np.ones((len(bufferINFO),len(bufferINFO[0])))
                    for i in range(0,len(bufferINFO)):
                        for j in range(0,len(bufferINFO[i])):
                            x[i,j]=bufferINFO[i][j,0]
                            y[i,j]=bufferINFO[i][j,1]
        
                    #绘制x-y图像
                    pd.options.display.notebook_repr_html=False  # 表格显示
                    plt.rcParams['figure.dpi'] = 100  # 图形分辨率
                    sns.set_theme(style='darkgrid')  # 图形主题

                    df=pd.DataFrame(dict(X=x[:,0],Y=y[:,0]))
                    sns.scatterplot(x=df['X'], y=df['Y'])
                    plt.plot(x[:,0], y[:,0],label='Leader')

                    for i in range(1,x.shape[1]):
                        df=pd.DataFrame(dict(X=x[:,i],Y=y[:,i]))
                        sns.scatterplot(x=df['X'], y=df['Y'])
                        plt.plot(x[:,i], y[:,i],label='Follower%d'%i)

                    plt.legend()
                    #plt.show()
                    plt.savefig('./Flocking_Results/flocking'+str(time_step)+ '.svg', format='svg')
                    plt.clf()
                    bufferINFO = []
            if done == True:
                s = self.env.reset()

                if len(bufferINFO)!=0:
                    bufferINFO=np.array(bufferINFO)
                    np.save(data_path + '/' + 'flocking' + str(time_step), bufferINFO)
                    x=np.ones((len(bufferINFO),len(bufferINFO[0])))
                    y=np.ones((len(bufferINFO),len(bufferINFO[0])))
                    for i in range(0,len(bufferINFO)):
                        for j in range(0,len(bufferINFO[i])):
                            x[i,j]=bufferINFO[i][j,0]
                            y[i,j]=bufferINFO[i][j,1]
        
                    #绘制x-y图像
                    pd.options.display.notebook_repr_html=False  # 表格显示
                    plt.rcParams['figure.dpi'] = 100  # 图形分辨率
                    sns.set_theme(style='darkgrid')  # 图形主题

                    df=pd.DataFrame(dict(X=x[:,0],Y=y[:,0]))
                    sns.scatterplot(x=df['X'], y=df['Y'])
                    plt.plot(x[:,0], y[:,0],label='Leader')

                    for i in range(1,x.shape[1]):
                        df=pd.DataFrame(dict(X=x[:,i],Y=y[:,i]))
                        sns.scatterplot(x=df['X'], y=df['Y'])
                        plt.plot(x[:,i], y[:,i],label='Follower%d'%i)
                        #print(x[:,i])
                    plt.legend()
                    #plt.show()
                    plt.savefig('./Flocking_Results/flocking'+str(time_step)+ '.svg', format='svg', bbox_inches='tight')
                    plt.clf()
                    bufferINFO = []
            u = []
            actions = []
            with torch.no_grad():
                for agent_id, agent in enumerate(self.agents):
                    #print('length:',len(s))
                    action = agent.select_action(s[agent_id], self.noise, self.epsilon)
                    u.append(action)
                    actions.append(action)
            for i in range(self.args.n_agents, self.args.n_players):
                actions.append([0, np.random.rand() * 2 - 1, 0, np.random.rand() * 2 - 1, 0])
            
            s_next, r, done, info = self.env.step(actions)
            info=np.array(info)
            bufferINFO.append(info)
            #print(info)
            self.buffer.store_episode(s[:self.args.n_agents], u, r[:self.args.n_agents], s_next[:self.args.n_agents])
            s = s_next
            if self.buffer.current_size >= self.args.batch_size:
                transitions = self.buffer.sample(self.args.batch_size)
                for agent in self.agents:
                    other_agents = self.agents.copy()
                    other_agents.remove(agent)
                    agent.learn(transitions, other_agents)
            if time_step > 0 and time_step % self.args.evaluate_rate == 0:
                returns.append(self.evaluate())
                np.save(reward_path + '/' + 'flocking' + str(time_step), returns)
                plt.figure()
                plt.plot(range(len(returns)), returns)
                plt.xlabel('episode * ' + str(self.args.evaluate_rate / self.episode_limit))
                plt.ylabel('average returns')
                plt.savefig(self.save_path + '/plt.png', format='png')
            self.noise = max(0.05, self.noise - 0.0000005)
            self.epsilon = max(0.05, self.epsilon - 0.0000005)
            #np.save(self.save_path + '/returns.pkl', returns)
            plt.clf()
            plt.close('all')

    def evaluate(self):
        returns = []
        for episode in range(self.args.evaluate_episodes):
            # reset the environment
            s = self.env.reset()
            rewards = 0
            for time_step in range(self.args.evaluate_episode_len):
                #self.env4.render()
                actions = []
                with torch.no_grad():
                    for agent_id, agent in enumerate(self.agents):
                        action = agent.select_action(s[agent_id], 0, 0)
                        actions.append(action)
                for i in range(self.args.n_agents, self.args.n_players):
                    actions.append([0, np.random.rand() * 2 - 1, 0, np.random.rand() * 2 - 1, 0])
                s_next, r, done, info = self.env.step(actions)
                rewards += r[0]
                s = s_next
            returns.append(rewards)
            print('Returns is', rewards)
        return sum(returns) / self.args.evaluate_episodes
