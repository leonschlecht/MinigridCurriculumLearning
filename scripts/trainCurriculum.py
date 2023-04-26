from datetime import datetime
from gymnasium.envs.registration import register

import utils
from curricula import linear, adaptive, RollingHorizonEvolutionaryAlgorithm


def main():
    args = utils.initializeArgParser()
    # TODO load time or set initially in linear / adaptive
    # TODO refactor to some utils method (for all methods)

    txtLogger = utils.get_txt_logger(utils.get_model_dir(args.model))

    # TODO fix iterations (so it doesnt overshoot the amount; maybe calculate with exact frame nrs or use updates)
    startTime: datetime = datetime.now()

    ############

    if args.trainEvolutionary:
        e = RollingHorizonEvolutionaryAlgorithm(txtLogger, startTime, args)
    elif args.trainAdaptive:
        adaptive.adaptiveCurriculum(txtLogger, startTime, args)

    elif args.trainLinear:
        linear.startLinearCurriculum(txtLogger, startTime, args)


def registerEnvs():
    # TODO find better way so that the max_steps decrease over time
    register(
        id="MiniGrid-Empty-Random-8x8-v0",  # todo use from ENV_NAMES
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


if __name__ == "__main__":
    registerEnvs()
    main()
