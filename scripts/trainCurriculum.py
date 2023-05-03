import sys
from datetime import datetime
from gymnasium.envs.registration import register

import utils
from curricula import linearCurriculum, RollingHorizonEvolutionaryAlgorithm, \
    BiasedRandomRollingHorizon, adaptiveCurriculum


def main():
    cmdLineString = ' '.join(sys.argv)
    args = utils.initializeArgParser()
    # TODO refactor to some utils method (for all methods)

    txtLogger = utils.get_txt_logger(
        utils.get_model_dir(args.model))  # TODO this is not clear if it creates a folder or not

    # TODO refactor scripts folder (so it there are actually only scripts in it)
    startTime: datetime = datetime.now()

    ############

    if args.trainEvolutionary:
        e = RollingHorizonEvolutionaryAlgorithm(txtLogger, startTime, cmdLineString, args)
    elif args.trainBiasedRandomRH:
        e = BiasedRandomRollingHorizon(txtLogger, startTime, args)
    elif args.trainRandomRH:
        e = BiasedRandomRollingHorizon(txtLogger, startTime, args)
    elif args.trainLinear:
        linearCurriculum.startLinearCurriculum(txtLogger, startTime, args)
    elif args.trainAdaptive:
        adaptiveCurriculum.startAdaptiveCurriculum(txtLogger, startTime, args)
    else:
        print("No training method selected!")


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
