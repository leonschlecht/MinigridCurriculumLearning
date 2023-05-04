import utils
from utils import initializeArgParser
import os
import json
from curricula import startAdaptiveCurriculum, BiasedRandomRollingHorizon, RollingHorizonEvolutionaryAlgorithm, linearCurriculum


def evaluateEvolutionaryCurriculum():
    evolutionary.evaluateCurriculumResults(trainingInfoJson)


def evaluateAdaptiveCurriculum():
    adaptive.evaluateAdaptiveCurriculum(trainingInfoJson)


def evaluateLinearCurriculum():
    linear.evaluateModel(trainingInfoJson)


if __name__ == "__main__":
    args = initializeArgParser()
    logFilePath = os.getcwd() + "\\storage\\" + args.model + "\\status.json"
    txtLogger = utils.get_txt_logger(utils.get_model_dir(args.model))

    if os.path.exists(logFilePath):
        with open(logFilePath, 'r') as f:
            trainingInfoJson = json.loads(f.read())
        trainingMethod = trainingInfoJson["trainMethod"]
        if trainingMethod == "evoltuionary":
            evaluateEvolutionaryCurriculum()
        elif trainingMethod == "linear":
            evaluateLinearCurriculum()
        elif trainingMethod == "adaptive":
            evaluateAdaptiveCurriculum()
    else:
        print("Model doesnt exist!")
    print(1)
    # Given a model name (which should probably have the trained method in it as well)
