import re
import time
import torch
import json

import utils
from training.ParallelEnvironment import MyParallelEnv
from utils import device, getEnvListThroughDifficulty


def startEvaluationInOneEnv(args, model, evalEnv, txtLogger) -> dict:
    # TODO decide if using args.argmax or not for evaluation
    # Load environments
    envs = []
    for i in range(args.procs):
        env = utils.make_env(evalEnv, args.seed + 10000 * i)
        envs.append(env)
    env = MyParallelEnv(envs)

    # Load agent
    model_dir = utils.get_model_dir(model)
    agent = utils.Agent(env.observation_space, env.action_space, model_dir,
                        argmax=args.argmax, num_envs=args.procs,
                        use_memory=args.memory, use_text=args.text)

    # Initialize logs
    logs = {"num_frames_per_episode": [], "return_per_episode": []}

    # Run agent
    start_time = time.time()
    obss = env.reset()

    log_done_counter = 0
    log_episode_return = torch.zeros(args.procs, device=device)
    log_episode_num_frames = torch.zeros(args.procs, device=device)

    while log_done_counter < args.episodes:
        actions = agent.get_actions(obss)
        obss, rewards, terminateds, truncateds, _ = env.step(actions)
        dones = tuple(a | b for a, b in zip(terminateds, truncateds))
        agent.analyze_feedbacks(rewards, dones)

        log_episode_return += torch.tensor(rewards, device=device, dtype=torch.float)
        log_episode_num_frames += torch.ones(args.procs, device=device)

        for i, done in enumerate(dones):
            if done:
                log_done_counter += 1
                logs["return_per_episode"].append(log_episode_return[i].item())
                logs["num_frames_per_episode"].append(log_episode_num_frames[i].item())

        mask = 1 - torch.tensor(dones, device=device, dtype=torch.float)
        log_episode_return *= mask
        log_episode_num_frames *= mask

    end_time = time.time()

    # Print logs
    num_frames = sum(logs["num_frames_per_episode"])
    evalTime = end_time - start_time
    fps = num_frames / evalTime
    return_per_episode = utils.synthesize(logs["return_per_episode"])
    num_frames_per_episode = utils.synthesize(logs["num_frames_per_episode"])
    formatted = "EVAL: {} with {} : F {} | FPS {:.0f} | duration {} | R:msmM {:.2f} {:.2f} {:.2f} {:.2f} | F:msmM {:.1f} {:.1f} {} {}".format(
        evalEnv, model, num_frames, fps, evalTime, *return_per_episode.values(), *num_frames_per_episode.values())
    txtLogger.info(formatted)

    evaluationResult = {
        "meanRet": return_per_episode["mean"],
        "maxRet": return_per_episode["max"],
        "minRet": return_per_episode["min"]
    }

    env.reset()
    return evaluationResult


def evaluateAll(model, envs, args, txtLogger) -> dict:
    utils.seed(args.seed)
    results = {"model": model}
    for evaluationEnv in envs:
        results[evaluationEnv] = startEvaluationInOneEnv(args, model, evaluationEnv, txtLogger)
    with open('storage/' + model + '/' + 'evaluation.json', 'w') as f:  # TODO use utils/storage file
        f.write(json.dumps(results, indent=4))
        # TODO check if this is even useful anymore and not already covered by other logfile
    txtLogger.info(f"Evaluation of {model} succeeded")
    return results


def getRewardMultiplier(evalEnv):
    """

    :param evalEnv:
    :return:
    """
    pattern = r'\d+'
    match = re.search(pattern, evalEnv)
    if match:
        return int(match.group())
    raise Exception("Something went wrong with the evaluation reward multiplier!", evalEnv)


def getDifficultyMultiplier(difficulty):
    if difficulty == 0:
        return 1
    elif difficulty == 1:
        return 1.1
    elif difficulty == 2:
        return 1.2
    raise Exception("Something went wrong with the difficulty multiplier! input difficulty:", difficulty)


def evaluateAgent(model, difficulty, args, txtLogger) -> int:
    """
    Evaluates and calculates the average performance in ALL environments
    Called from other classes to start the evaluation
    :param txtLogger:
    :param model: the name of the model
    :param difficulty:
    :param args: the command line arugments
    :return: the average reward
    """
    rewardSum = 0
    envs = getEnvListThroughDifficulty(difficulty)
    evaluationResult = evaluateAll(model, envs, args, txtLogger)
    for evalEnv in envs:
        currentReward = float(evaluationResult[evalEnv]["meanRet"]) * getRewardMultiplier(evalEnv)
        rewardSum += currentReward
    print("Evaluate agent TEST", rewardSum * getDifficultyMultiplier(difficulty))
    return rewardSum * getDifficultyMultiplier(difficulty)
