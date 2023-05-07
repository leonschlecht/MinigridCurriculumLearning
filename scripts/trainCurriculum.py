import sys
from datetime import datetime

import numpy as np
from gymnasium.envs.registration import register

import utils
from curricula import linearCurriculum, RollingHorizonEvolutionaryAlgorithm, \
    adaptiveCurriculum, RandomRollingHorizon
from utils import ENV_NAMES


def main():
    cmdLineString = ' '.join(sys.argv)
    args = utils.initializeArgParser()
    # TODO add --debug option with some preset parameters, and only use more params if != default ones

    txtLogger = utils.get_txt_logger(
        utils.get_model_dir(args.model))  # TODO this is not clear if it creates a folder or not

    # TODO refactor scripts folder (so it there are actually only scripts in it)
    startTime: datetime = datetime.now()

    ############
    # TODO make e.start() methods instaed of doing it in init because of calling eval later
    if args.trainEvolutionary:
        e = RollingHorizonEvolutionaryAlgorithm(txtLogger, startTime, cmdLineString, args)
    elif args.trainBiasedRandomRH:
        e = RandomRollingHorizon(txtLogger, startTime, cmdLineString, args, False)
    elif args.trainRandomRH:
        e = RandomRollingHorizon(txtLogger, startTime, cmdLineString, args, True)
    elif args.trainLinear:
        linearCurriculum.startLinearCurriculum(txtLogger, startTime, args)
    elif args.trainAdaptive:
        adaptiveCurriculum.startAdaptiveCurriculum(txtLogger, startTime, args)
    else:
        print("No training method selected!")


def registerEnvs():
    """
    Registers the envs before the training. Each env has 3 difficulty settings, whereby they decrease their maxsteps
    """
    """
    register(
        id="MiniGrid-Empty-Random-8x8-v0",
        entry_point="minigrid.envs:EmptyEnv",
        kwargs={"size": 8, "agent_start_pos": None},
    ) """

    maxSteps16x16 = 16 ** 2 * 10
    maxSteps8x8 = 8 ** 2 * 10
    maxSteps6x6 = 6 ** 2 * 10
    maxSteps5x5 = 5 ** 2 * 10
    maxSteps = np.array([maxSteps5x5, maxSteps6x6, maxSteps8x8, maxSteps16x16])
    difficulty = np.array([1, 0.33, 0.11])
    result = np.round(np.matmul(maxSteps.reshape(-1, 1), difficulty.reshape(1, -1)))

    # TODO numpy.int32 vs int
    for i in range(len(difficulty)):
        register(
            id=ENV_NAMES.DOORKEY_16x16 + ENV_NAMES.CUSTOM_POSTFIX + str(i),
            entry_point="minigrid.envs:DoorKeyEnv",
            kwargs={"size": 16, "max_steps": int(result[3][i])},
        )

        register(
            id=ENV_NAMES.DOORKEY_8x8 + ENV_NAMES.CUSTOM_POSTFIX + str(i),
            entry_point="minigrid.envs:DoorKeyEnv",
            kwargs={"size": 8, "max_steps": int(result[2][i])},
        )

        register(
            id=ENV_NAMES.DOORKEY_6x6 + ENV_NAMES.CUSTOM_POSTFIX + str(i),
            entry_point="minigrid.envs:DoorKeyEnv",
            kwargs={"size": 6, "max_steps": int(result[1][i])},
        )

        register(
            id=ENV_NAMES.DOORKEY_5x5 + ENV_NAMES.CUSTOM_POSTFIX + str(i),
            entry_point="minigrid.envs:DoorKeyEnv",
            kwargs={"size": 5, "max_steps": int(result[0][i])},
        )

    # TODO vllt ist register gar nicht so teuer/umständlich, so dass man das einfach exakt je nach performance berechnen kann



if __name__ == "__main__":
    registerEnvs()
    main()
