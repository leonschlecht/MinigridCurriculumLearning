import utils
from utils import ENV_NAMES
from curricula import linear, adaptive, EvolutionaryCurriculum
import time
import numpy as np

if __name__ == "__main__":
    args = utils.initializeArgParser()
    # TODO load time or set initially in linear / adaptive
    # TODO refactor to some utils method (for all methods)

    txtLogger = utils.get_txt_logger(utils.get_model_dir(args.model))

    # TODO limit max frames per env in evaluation
    # TODO fix iterations (so it doesnt overshoot the amount; maybe calculate with exact frame nrs or use updates)
    ITERATIONS_PER_ENV = args.iterationsPerEnv
    startTime = time.time()

    if args.trainEvolutionary:
        e = EvolutionaryCurriculum(txtLogger, startTime, args)

    if args.trainAdaptive:
        adaptive.adaptiveCurriculum(txtLogger, startTime, args)

    if args.trainLinear:
        linear.startLinearCurriculum(txtLogger, startTime, args)
