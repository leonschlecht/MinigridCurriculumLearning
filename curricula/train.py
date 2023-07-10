import os
import time
import tensorboardX

import utils
from training.PPO import MyPPOAlgo
from utils import device
from model import ACModel


def startTraining(framesToTrain: int, currentFramesDone, model: str, envList: list, args, txt_logger) -> int:
    """
    :param currentFramesDone:
    :param txt_logger: reference to the .txt log file
    :param framesToTrain: the number of iterations
    :param model: name of the model - where the training will be saved
    :param envList: list of the name of the environments to be trained on
    :param args: the command lines arguments that get parsed and passed through
    :return: the exact number of iterations done
    """
    # TODo split this into multiple methods maybe
    model_name = model
    model_dir = utils.get_model_dir(model_name)

    utils.seed(args.seed)
    # Load environments
    envs = []
    for i in range(args.procs // len(envList)):
        for j in range(len(envList)):
            envs.append(utils.make_env(envList[j], args.seed + 10000 * (i * len(envList) + j)))

    assert len(envs) == args.procs, f"Length of envs {len(envs)} is not equal to amount of processes {args.procs}"
    assert args.procs % args.paraEnv == 0, \
        "The amount of processes must be divisble by the amount of envs to be trained on in parallel" 

    # Load training status
    try:
        status = utils.get_status(model_dir)  # TODO fix try except
    except OSError:
        status = {"num_frames": 0, "update": 0}

    # Load observations preprocessor
    obs_space, preprocess_obss = utils.get_obss_preprocessor(envs[0].observation_space)
    if "vocab" in status:
        preprocess_obss.vocab.load_vocab(status["vocab"])
    # txt_logger.info("Observations preprocessor loaded")

    # Load model
    acmodel = ACModel(obs_space, envs[0].action_space, args.mem, args.text)

    if "model_state" in status:
        acmodel.load_state_dict(status["model_state"])
    acmodel.to(device)

    # currentFramesDone = status["num_frames"]
    update = status["update"]
    start_time = time.time()
    framesWithThisEnv = 0

    if framesToTrain == 0:
        if not os.path.isdir(model_dir):
            os.mkdir(model_dir)
        else:
            raise Exception("Path exists when trying to create epoch0 folder")
        txt_logger.info(f'{acmodel}')
        txt_logger.info(f'Created model {model}')
        return 0
    algo = MyPPOAlgo(envs, acmodel, device, args.frames_per_proc, args.discount, args.lr, args.gae_lambda,
                     args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                     args.optim_eps, args.clip_eps, args.epochs, args.batch_size, preprocess_obss)

    # txt_logger.info(f"\tAlgorithm loaded in {round(-start_time + time.time(), 2)} sec")

    if "optimizer_state" in status:
        algo.optimizer.load_state_dict(status["optimizer_state"])
    duration = 0
    while currentFramesDone < framesToTrain:
        update_start_time = time.time()

        exps, logs1 = algo.collect_experiences()
        logs2 = algo.update_parameters(exps)
        logs = {**logs1, **logs2}
        update_end_time = time.time()

        framesWithThisEnv += logs["num_frames"]  # TODO can probably calculate this with end - startFrames

        currentFramesDone += logs["num_frames"]
        update += 1

        # Print logs
        if update % args.log_interval == 0:
            fps = logs["num_frames"] / (update_end_time - update_start_time)
            duration = int(time.time() - start_time)
            return_per_episode = utils.synthesize(logs["return_per_episode"])
            rreturn_per_episode = utils.synthesize(logs["reshaped_return_per_episode"])
            num_frames_per_episode = utils.synthesize(logs["num_frames_per_episode"])

            header = ["update", "framesToTrain", "FPS", "duration"]
            data = [update, currentFramesDone, fps, duration]
            header += ["rreturn_" + key for key in rreturn_per_episode.keys()]
            data += rreturn_per_episode.values()
            header += ["num_frames_" + key for key in num_frames_per_episode.keys()]
            data += num_frames_per_episode.values()
            header += ["entropy", "value", "policy_loss", "value_loss", "grad_norm"]
            data += [logs["entropy"], logs["value"], logs["policy_loss"], logs["value_loss"], logs["grad_norm"]]

            # txt_logger.info(
            #    "\t{} | {} | curF {} | U {} | AllF {:07} | FPS {:04.0f} | D {} | rR:msmM {:.3f} {:.2f} {:.2f} {:.2f} | F:msmM {:.1f} {:.1f} {} {} | H {:.2f} | V {:.4f} | pL {:.4f} | vL {:.4f} | g {:.4f}"
            #   .format(envList, model, framesWithThisEnv, *data))

            header += ["return_" + key for key in return_per_episode.keys()]
            data += return_per_episode.values()


        # Save status
        if update % args.save_interval == 0 or currentFramesDone >= framesToTrain:
            status = {"num_frames": currentFramesDone, "update": update,
                      "model_state": acmodel.state_dict(), "optimizer_state": algo.optimizer.state_dict()}
            if hasattr(preprocess_obss, "vocab"):
                status["vocab"] = preprocess_obss.vocab.vocab
            utils.save_status(status, model_dir)
            # txt_logger.info("\t\tStatus saved")

    txt_logger.info(f'\nTrained on {envList} using model {model} for {framesWithThisEnv} frames. '
                    f'Duration {time.time() - start_time}. Fps: ??. totalF {status["num_frames"]}')

    algo.env.end()
    algo.env.close()
    time.sleep(1)
    return status["num_frames"]
