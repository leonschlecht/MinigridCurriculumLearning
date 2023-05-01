import csv
import os
import torch
import logging
import sys
import shutil

import utils
from .other import device


def create_folders_if_necessary(path):
    dirname = os.path.dirname(path)
    if not os.path.isdir(dirname):
        os.makedirs(dirname)


def get_storage_dir():
    if "RL_STORAGE" in os.environ:
        return os.environ["RL_STORAGE"]
    return "storage"


def get_model_dir(model_name):
    return os.path.join(get_storage_dir(), model_name)


def get_status_path(model_dir):
    return os.path.join(model_dir, "status.pt")


def get_status(model_dir):
    path = get_status_path(model_dir)
    return torch.load(path, map_location=device)


def save_status(status, model_dir):
    path = get_status_path(model_dir)
    utils.create_folders_if_necessary(path)
    torch.save(status, path)


def get_vocab(model_dir):
    return get_status(model_dir)["vocab"]


def get_model_state(model_dir):
    return get_status(model_dir)["model_state"]


def get_txt_logger(model_dir):
    path = os.path.join(model_dir, "log.txt")
    utils.create_folders_if_necessary(path)

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(filename=path),
            logging.StreamHandler(sys.stdout)
        ]
    )
    txtLogger = logging.getLogger()
    txtLogger.info(f"Device: {device}")
    return txtLogger


def get_csv_logger(model_dir):
    csv_path = os.path.join(model_dir, "log.csv")
    utils.create_folders_if_necessary(csv_path)
    csv_file = open(csv_path, "a")
    return csv_file, csv.writer(csv_file)


###

def getModelWithCurricGenSuffix(model, curriculumNr: int, genNr: int) -> str:
    """

    :param model:
    :param curriculumNr:
    :param genNr:
    :return:
    """
    return model + "_curric" + str(curriculumNr) + '_gen' + str(genNr)


def getModelWithCurricSuffix(model, epoch, curricNr) -> str:
    return getEpochModelName(model, epoch) + "_curric" + str(curricNr)


def getEpochModelName(model, epoch) -> str:
    return model + "\\epoch_" + str(epoch)


def getModelWithCandidatePrefix(model) -> str:
    """

    :param model:
    :return:
    """
    return model + "_CANDIDATE"


def copyAgent(src, dest) -> None:
    """

    :param src:
    :param dest:
    :return:
    """
    pathPrefix = os.getcwd() + '\\storage\\'
    fullSrcPath = pathPrefix + src
    fullDestPath = pathPrefix + dest
    if os.path.isdir(fullDestPath):
        raise Exception(f"Path exists at {fullDestPath}! Copying agent failed")
    else:
        shutil.copytree(fullSrcPath, fullDestPath)
        print(f'Copied Agent! {src} ---> {dest}')


def deleteModelIfExists(directory) -> bool:
    """
    Deletes a path if it exists. Returns true on success, false otherwise
    :param directory: name of the model to be deleted, which is stored in /storage
    """
    fullPath = os.getcwd() + "\\storage\\" + directory  # TODO use os.join
    if os.path.exists(fullPath):  # TODO split this into 2 methods
        shutil.rmtree(fullPath)
        return True
    return False
