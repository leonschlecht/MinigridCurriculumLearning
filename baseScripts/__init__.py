from gymnasium.envs.registration import register

register(
    id="MiniGrid-Empty-Random-8x8-v0",
    entry_point="minigrid.envs:EmptyEnv",
    kwargs={"size": 8, "agent_start_pos": None},
)

register(
    id="MiniGrid-Empty-Random-16x16-v0",
    entry_point="minigrid.envs:EmptyEnv",
    kwargs={"size": 16, "agent_start_pos": None},
)

register(
    id="MiniGrid-DoorKey-12x12-v0",
    entry_point="minigrid.envs:DoorKeyEnv",
    kwargs={"size": 12},
)
