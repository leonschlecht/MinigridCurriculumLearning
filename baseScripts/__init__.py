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
    id="MiniGrid-DoorKey-16x16-custom",
    entry_point="minigrid.envs:DoorKeyEnv",
    kwargs={"size": 16, "max_steps": 300},
)

register(
    id="MiniGrid-DoorKey-8x8-custom",
    entry_point="minigrid.envs:DoorKeyEnv",
    kwargs={"size": 8, "max_steps": 200},
)

register(
    id="MiniGrid-DoorKey-6x6-custom",
    entry_point="minigrid.envs:DoorKeyEnv",
    kwargs={"size": 6, "max_steps": 125},
)

register(
    id="MiniGrid-DoorKey-5x5-custom",
    entry_point="minigrid.envs:DoorKeyEnv",
    kwargs={"size": 5, "max_steps": 100},
)
