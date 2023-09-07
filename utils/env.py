import gymnasium as gym
from Minigrid.minigrid.wrappers import ViewSizeWrapper


def make_env(env_key, seed=None, render_mode=None):
    """
    Creates an environment given the name and seed.
    Transforms it with the viewsize wrapper at the end
    """
    env = gym.make(env_key, render_mode=render_mode)  # max_episode_steps ?? ; can potentially call kwargs
    env.reset(seed=seed)
    return ViewSizeWrapper(env, agent_view_size=5)
