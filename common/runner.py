import os
import torch
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from common.agent import Agent
from matplotlib import pyplot as plt
from common.replay_buffer import Buffer

matplotlib.use('Agg')
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class Runner:
    def __init__(self, args, env):
        self.args = args
        self.noise = args.noise_rate
        self.epsilon = args.epsilon
        self.episode_limit = args.max_episode_len
        self.max_episodes = args.max_episodes
        self.env = env
        self.worker_num = args.worker_num
        self.agents = self._init_agents()
        self.buffer = Buffer(args)
        self.check_path()
        print('Starting processor %d \n' % self.worker_num)

    def _init_agents(self):
        agents = []
        for i in range(self.args.n_agents):
            agent = Agent(i, self.args, self.worker_num)
            agents.append(agent)

        return agents

    def run(self):
        returns = []
        # Data store
        bufferINFO = []
        done = False
        for time_step in tqdm(range(self.args.time_steps)):
            if time_step % self.episode_limit == 0 or done == True:
                if time_step % self.args.evaluate_rate == 0 and time_step > 0:
                    returns.append(self.evaluate())
                    np.save(os.path.join(self.reward_path, 'flocking' + str(time_step)), returns)
                    self.plot_learn_curve(returns)
                    self.plot_xOy(bufferINFO, time_step)
                s = self.env.reset()
                bufferINFO = []
                done = False
            with torch.no_grad():
                # action batch
                u = [agent.select_action(s[agent_id], self.noise, self.epsilon)
                     for agent_id, agent in enumerate(self.agents)]
                actions = u
            for i in range(self.args.n_agents, self.args.n_players):
                actions.append([0, np.random.rand() * 2 - 1, 0, np.random.rand() * 2 - 1, 0])
            actions = np.array(actions, dtype=np.float64)
            s_next, r, done, info = self.env.step(actions)
            bufferINFO.append(np.array(info))
            self.buffer.store_episode(s[:self.args.n_agents], u, r[:self.args.n_agents],
                                      s_next[:self.args.n_agents])
            s = s_next
            if self.buffer.current_size >= self.args.batch_size:
                transitions = self.buffer.sample(self.args.batch_size)
                for agent in self.agents:
                    other_agents = self.agents.copy()
                    other_agents.remove(agent)
                    agent.learn(transitions, other_agents)

    def evaluate(self):
        returns = []
        self.env.args.evaluate = True
        for episode in range(self.args.evaluate_episodes):
            # reset the environment
            s = self.env.reset()
            rewards = 0
            for time_step in range(self.args.evaluate_episode_len):
                actions = []
                with torch.no_grad():
                    for agent_id, agent in enumerate(self.agents):
                        action = agent.select_action(s[agent_id], 0, 0)
                        actions.append(action)
                for i in range(self.args.n_agents, self.args.n_players):
                    actions.append([0, np.random.rand() * 2 - 1, 0, np.random.rand() * 2 - 1, 0])
                actions = np.array(actions, dtype=np.float64)
                s_next, r, done, info = self.env.step(actions)
                rewards += r[0] / self.args.evaluate_episodes
                s = s_next
            returns.append(rewards)
        print('\nReturns is: %s' % max(returns))
        self.env.args.evaluate = False
        return sum(returns)

    def check_path(self):
        self.save_path = os.path.join(self.args.save_dir, '%s-%s-agents-thread-%s' % (
        self.args.scenario_name, self.args.n_agents, self.worker_num))
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        self.data_path = './data/flocking data/%s-agents-thread-%s/' % (self.args.n_agents, self.worker_num)
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)
        self.reward_path = './data/reward data/%s-agents-thread-%s/' % (self.args.n_agents, self.worker_num)
        if not os.path.exists(self.reward_path):
            os.makedirs(self.reward_path)
        self.flocking_path = './Flocking_Results/%s-agents-thread-%s/' % (self.args.n_agents, self.worker_num)
        if not os.path.exists(self.flocking_path):
            os.makedirs(self.flocking_path)

    def plot_xOy(self, bufferINFO, time_step):
        bufferINFO = np.array(bufferINFO)
        np.save(self.data_path + '/' + 'flocking%s' % (time_step), bufferINFO)
        x = np.ones((len(bufferINFO), len(bufferINFO[0])))
        y = np.ones((len(bufferINFO), len(bufferINFO[0])))
        for i in range(0, len(bufferINFO)):
            for j in range(0, len(bufferINFO[i])):
                x[i, j] = bufferINFO[i][j, 0]
                y[i, j] = bufferINFO[i][j, 1]
        pd.options.display.notebook_repr_html = False
        plt.rcParams['figure.dpi'] = 350
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['axes.labelsize'] = 14
        plt.rcParams['xtick.labelsize'] = 14
        plt.rcParams['ytick.labelsize'] = 14
        plt.rcParams['font.size'] = 14
        sns.set_theme(style='darkgrid')
        df = pd.DataFrame(dict(X=x[:, 0], Y=y[:, 0]))
        sns.scatterplot(x=df['X'], y=df['Y'])
        plt.plot(x[:, 0], y[:, 0], label='Leader')
        for i in range(1, x.shape[1]):
            df = pd.DataFrame(dict(X=x[:, i], Y=y[:, i]))
            sns.scatterplot(x=df['X'], y=df['Y'])
            plt.plot(x[:, i], y[:, i], label='Follower%d' % i)
        plt.legend(prop={'family': 'Times New Roman', 'size': 14})
        plt.xlabel('x (m)', fontsize=14, fontname='Times New Roman')
        plt.ylabel('y (m)', fontsize=14, fontname='Times New Roman')
        plt.xticks(fontname='Times New Roman', fontsize=14)
        plt.yticks(fontname='Times New Roman', fontsize=14)
        plt.savefig(os.path.join(self.flocking_path, 'flocking' + str(time_step) + '.svg'), format='svg')
        plt.clf()
        plt.close('all')

    def plot_learn_curve(self, returns):
        pd.options.display.notebook_repr_html = False
        plt.rcParams['figure.dpi'] = 350
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['font.size'] = 14
        sns.set_theme(style='darkgrid')
        plt.figure()
        plt.plot(range(len(returns)), returns, label='%s-follower' % self.args.n_agents)
        plt.xlabel('Episode')
        plt.ylabel('Returns')
        plt.legend()
        plt.savefig(os.path.join(self.save_path, 'plt.png'), format='png')
        plt.clf()
        plt.close('all')
