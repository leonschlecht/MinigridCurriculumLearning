import gymnasium as gym


def make_env(env_key, seed=None, render_mode=None):
    env = gym.make(env_key, render_mode=render_mode)
    env.reset(seed=seed)
    # env.observation_space = env_obs
    return ViewSizeWrapper(env, agent_view_size=5)
    """
    obs, _ = env.reset()
    print(obs["image"].shape)
    env_obs = ViewSizeWrapper(env, agent_view_size=5)
    obs, _ = env_obs.reset()
    print(obs["image"].shape)
    print("--------------------------------")
    """
    return env
