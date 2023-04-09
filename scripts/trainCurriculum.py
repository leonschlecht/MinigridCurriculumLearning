import utils
from utils import ENV_NAMES
from curricula import linear, adaptive, EvolutionaryCurriculum
import time


if __name__ == "__main__":
    args = utils.initializeArgParser()
    # TODO --individualNrs, --iterationsPerEnv
    # TODO load time or set initially
    # TODO refactor to some utils method (for all methods)

    txtLogger = utils.get_txt_logger(utils.get_model_dir(args.model))

    uniformCurriculum = [ENV_NAMES.DOORKEY_5x5, ENV_NAMES.DOORKEY_6x6, ENV_NAMES.DOORKEY_8x8, ENV_NAMES.DOORKEY_16x16]
    focus8 = [ENV_NAMES.DOORKEY_8x8, ENV_NAMES.DOORKEY_8x8, ENV_NAMES.DOORKEY_8x8, ENV_NAMES.DOORKEY_6x6]
    mix16_8 = [ENV_NAMES.DOORKEY_16x16, ENV_NAMES.DOORKEY_16x16, ENV_NAMES.DOORKEY_8x8, ENV_NAMES.DOORKEY_8x8]
    idk = [ENV_NAMES.DOORKEY_16x16, ENV_NAMES.DOORKEY_8x8, ENV_NAMES.DOORKEY_16x16, ENV_NAMES.DOORKEY_6x6]

    curricula = [
        uniformCurriculum,
        # focus8,
        # mix16_8,
        # idk
    ]

    ITERATIONS_PER_ENV = 150000
    PRE_TRAIN_FRAMES = ITERATIONS_PER_ENV
    startTime = time.time()
    if args.trainEvolutionary:
        e = EvolutionaryCurriculum(ITERATIONS_PER_ENV, txtLogger, startTime, curricula, args)

    if args.trainAdaptive:
        adaptive.adaptiveCurriculum(args, ITERATIONS_PER_ENV, txtLogger, startTime)

    if args.trainLinear:
        linear.startLinearCurriculum(args, startTime, txtLogger)
