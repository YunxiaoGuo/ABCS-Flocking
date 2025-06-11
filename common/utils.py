import numpy as np
import inspect
import functools
import sys

sys.path.append("..")
from envs import CS_flocking


def store_args(method):
    """Stores provided method args as instance attributes.
    """
    argspec = inspect.getfullargspec(method)
    defaults = {}
    if argspec.defaults is not None:
        defaults = dict(
            zip(argspec.args[-len(argspec.defaults):], argspec.defaults))
    if argspec.kwonlydefaults is not None:
        defaults.update(argspec.kwonlydefaults)
    arg_names = argspec.args[1:]

    @functools.wraps(method)
    def wrapper(*positional_args, **keyword_args):
        self = positional_args[0]
        # Get default arg values
        args = defaults.copy()
        # Add provided arg values
        for name, value in zip(arg_names, positional_args[1:]):
            args[name] = value
        args.update(keyword_args)
        self.__dict__.update(args)
        return method(*positional_args, **keyword_args)

    return wrapper


def make_env(args):
    env = CS_flocking(args.n_agents, args)
    args.n_players = env.n
    args.n_agents = env.n
    args.obs_shape = [env.observation_space[i].shape[0] for i in range(args.n_agents)]
    action_shape = []
    for content in env.action_space:
        action_shape.append(content.shape[0])
    args.action_shape = action_shape[:args.n_agents]
    args.high_action = 1
    args.low_action = -1
    return env, args
