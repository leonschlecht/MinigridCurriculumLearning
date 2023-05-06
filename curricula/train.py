import time
import torch_ac
import tensorboardX
import sys

import utils
from utils import device
from model import ACModel


def main(framesToTrain: int, currentFramesDone, model: str, env: str, args, txt_logger) -> int:
    """

    :param currentFramesDone:
    :param txt_logger: reference to the .txt log file
    :param framesToTrain: the number of iterations
    :param model: name of the model - where the training will be saved
    :param env: the name of the environment
    :param args: the command lines arguments that get parsed and passed through
    :return: the exact number of iterations done
    """
    # TODo split this into multiple methods maybe
    # Set run dir
    model_name = model
    model_dir = utils.get_model_dir(model_name)
    csv_file, csv_logger = utils.get_csv_logger(model_dir)
    tb_writer = tensorboardX.SummaryWriter(model_dir)

    # Log command and all script arguments
    # txt_logger.info("{}\n".format(" ".join(sys.argv))) # TODO use this
    # txt_logger.info("{}\n".format(args))

    # Set seed for all randomness sources
    utils.seed(args.seed)

    # Load environments
    envs = []
    for i in range(args.procs):
        envs.append(utils.make_env(env, args.seed + 10000 * i))  # TODO what does 10k mean here ?
    # txt_logger.info("Environments loaded\n")

    # Load training status
    try:
        status = utils.get_status(model_dir)  # TODO find better way than try except
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
        txt_logger.info(f'Created model {model}')
        return 0
    algo = torch_ac.PPOAlgo(envs, acmodel, device, args.frames_per_proc, args.discount, args.lr, args.gae_lambda,
                            args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                            args.optim_eps, args.clip_eps, args.epochs, args.batch_size, preprocess_obss)

    txt_logger.info(f"\tAlgorithm loaded in {round(-start_time + time.time(), 2)} sec")

    if "optimizer_state" in status:
        algo.optimizer.load_state_dict(status["optimizer_state"])

    print("done=", currentFramesDone, "toTrain=", framesToTrain)
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

            txt_logger.info(
                "\t{} | {} | curF {} | U {} | AllF {:07} | FPS {:04.0f} | D {} | rR:msmM {:.3f} {:.2f} {:.2f} {:.2f} | F:msmM {:.1f} {:.1f} {} {} | H {:.2f} | V {:.4f} | pL {:.4f} | vL {:.4f} | g {:.4f}"
                .format(env, model, framesWithThisEnv, *data))

            header += ["return_" + key for key in return_per_episode.keys()]
            data += return_per_episode.values()

            if status["num_frames"] == 0:
                csv_logger.writerow(header)
            csv_logger.writerow(data)
            csv_file.flush()

            for field, value in zip(header, data):
                tb_writer.add_scalar(field, value, currentFramesDone)

        # Save status
        if update % args.save_interval == 0 or currentFramesDone >= framesToTrain:
            status = {"num_frames": currentFramesDone, "update": update,
                      "model_state": acmodel.state_dict(), "optimizer_state": algo.optimizer.state_dict()}
            if hasattr(preprocess_obss, "vocab"):
                status["vocab"] = preprocess_obss.vocab.vocab
            utils.save_status(status, model_dir)
            # txt_logger.info("\t\tStatus saved")

    txt_logger.info(
        'Trained on' + env + ' using model ' + model + ' for ' + str(framesWithThisEnv) + ' frames')  # TODO f
    algo.env.close()
    tb_writer.close()
    return status["num_frames"]