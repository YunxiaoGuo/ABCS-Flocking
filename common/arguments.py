import argparse
import numpy as np
"""
Here are the param for the training
"""

def get_args():
    parser = argparse.ArgumentParser("Fixed-wing UAV Leader-Follower Flocking with MARL")
    parser.add_argument("--worker-num", type=int, default=10, help="Number of the worker in multiprocess")
    # Environment
    parser.add_argument("--scenario-name", type=str, default="l-f_flocking", help="name of the scenario script (distinguish different experiments)")
    parser.add_argument("--max-episode-len", type=int, default=120, help="maximum episode length")
    parser.add_argument("--max-episodes", type=int, default=10000, help="maximum episodes")
    parser.add_argument("--time-steps", type=int, default=1000000, help="number of time steps")
    parser.add_argument("--n-agents", type=int, default=5, help="number of flocking agents")
    parser.add_argument("--num-adversaries", type=int, default=1, help="number of adversaries")
    # Core training parameters
    parser.add_argument("--lr-actor", type=float, default=1e-4, help="learning rate of actor")
    parser.add_argument("--lr-critic", type=float, default=1e-3, help="learning rate of critic")
    parser.add_argument("--epsilon", type=float, default=0.1, help="epsilon greedy")
    parser.add_argument("--noise_rate", type=float, default=0.1, help="noise rate for sampling from a standard normal distribution ")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--tau", type=float, default=0.01, help="parameter for updating the target network")
    parser.add_argument("--beta", type=float, default=1, help="hyperparameter beta for Leader-Guided Cucker-Smale Reward")
    parser.add_argument("--sigma", type=float, default=10, help="hyperparameter sigma for Leader-Guided Cucker-Smale Reward")
    parser.add_argument("--theta", type=float, default=1,help="hyperparameter theta for Leader-Guided Cucker-Smale Reward")
    parser.add_argument("--Cr", type=float, default=1,help="hyperparameter Cr for Leader-Guided Cucker-Smale Reward")
    parser.add_argument("--Cv", type=float, default=3,help="hyperparameter Cv for Leader-Guided Cucker-Smale Reward")
    # UAV shape parameters
    parser.add_argument("--wing-length", type=float, default=3, help="wing length of the fixed-wing UAV")
    parser.add_argument("--air-length", type=float, default=3, help="air length of the fixed-wing UAV")
    parser.add_argument("--vmax", type=float, default=3, help="maximum delta v in each update of the fixed-wing UAV")
    parser.add_argument("--rollmax", type=float, default=np.pi/18, help="maximum delta roll in each update of the fixed-wing UAV")
    # Buffer
    parser.add_argument("--buffer-size", type=int, default=int(5e5), help="number of transitions can be stored in buffer")
    parser.add_argument("--batch-size", type=int, default=512, help="number of episodes to optimize at the same time")
    # Checkpointing
    parser.add_argument("--save-dir", type=str, default="./model", help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=2000, help="save model once every time this many episodes are completed")
    parser.add_argument("--model-dir", type=str, default="", help="directory in which training state and model are loaded")
    # Evaluate
    parser.add_argument("--evaluate-episodes", type=int, default=100, help="number of episodes for evaluating")
    parser.add_argument("--evaluate-episode-len", type=int, default=120, help="length of episodes for evaluating")
    parser.add_argument("--evaluate", type=bool, default=False, help="whether to evaluate the model")
    parser.add_argument("--evaluate-rate", type=int, default=2400, help="how often to evaluate model")
    args = parser.parse_args()
    return args
