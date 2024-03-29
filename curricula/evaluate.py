import time
import torch
import json

import utils
from training.ParallelEnvironment import MyParallelEnv
from utils import device, getEnvListThroughDifficulty
from utils.curriculumHelper import getRewardMultiplier


def startEvaluationInOneEnv(args, model, evalEnv, txtLogger) -> dict:
    """
    Execute evaluation in a single env, and return the results dictionary
    """
    # Load environments
    envs = []
    if args.procs > args.episodes:
        envsToLoad = args.episodes
    else:
        envsToLoad = args.procs
    for i in range(envsToLoad):
        env = utils.make_env(evalEnv, args.seed + 10000 * i)
        envs.append(env)
    env = MyParallelEnv(envs)

    # Load agent
    model_dir = utils.get_model_dir(model)
    agent = utils.Agent(env.observation_space, env.action_space, model_dir,
                        argmax=args.argmax, num_envs=envsToLoad,
                        use_memory=args.memory, use_text=args.text)

    # Initialize logs
    logs = {"num_frames_per_episode": [], "return_per_episode": []}

    # Run agent
    start_time = time.time()
    obss = env.reset()

    log_done_counter = 0
    log_episode_return = torch.zeros(envsToLoad, device=device)
    log_episode_num_frames = torch.zeros(envsToLoad, device=device)

    while log_done_counter < args.episodes:
        actions = agent.get_actions(obss)
        obss, rewards, terminateds, truncateds, _ = env.step(actions)
        dones = tuple(a | b for a, b in zip(terminateds, truncateds))
        agent.analyze_feedbacks(rewards, dones)

        log_episode_return += torch.tensor(rewards, device=device, dtype=torch.float)
        log_episode_num_frames += torch.ones(envsToLoad, device=device)

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
    evalTime = end_time - start_time + .0001
    fps = num_frames / evalTime
    return_per_episode = utils.synthesize(logs["return_per_episode"])
    num_frames_per_episode = utils.synthesize(logs["num_frames_per_episode"])
    formatted = "\tEVAL: {} with {} : F {} | FPS {:.0f} | duration {} | R:msmM {:.2f} {:.2f} {:.2f} {:.2f} | F:msmM {:.1f} {:.1f} {} {}".format(
        evalEnv, model, num_frames, fps, evalTime, *return_per_episode.values(), *num_frames_per_episode.values())
    # txtLogger.info(formatted)

    evaluationResult = {
        "meanRet": return_per_episode["mean"],
        "maxRet": return_per_episode["max"],
        "minRet": return_per_episode["min"]
    }

    env.reset()
    env.end()
    env.close()
    return evaluationResult


def evaluateAll(model, envs, args, txtLogger) -> dict:
    """
    Executes the evaluation in all envs and returns the full results dictionary
    """
    utils.seed(args.seed)
    results = {"model": model}
    for evaluationEnv in envs:
        results[evaluationEnv] = startEvaluationInOneEnv(args, model, evaluationEnv, txtLogger)
    with open('storage/' + model + '/' + 'evaluation.json', 'w') as f:  # TODO use utils/storage file
        f.write(json.dumps(results, indent=4))
        # TODO check if this is even useful anymore and not already covered by other logfile
    # txtLogger.info(f"Evaluation of {model} succeeded")
    return results


def evaluateAgent(model, difficulty, args, txtLogger, envList: list) -> list:
    """
    Evaluates and calculates the average performance in ALL environments
    Called from other classes to start the evaluation
    :param envList:
    :param txtLogger:
    :param model: the name of the model
    :param difficulty:
    :param args: the command line arugments
    :return: the average reward
    """
    startTime = time.time()
    rewards = []
    envs = getEnvListThroughDifficulty(difficulty, envList)
    evaluationResult = evaluateAll(model, envs, args, txtLogger)
    for evalEnv in envs:
        currentReward = float(evaluationResult[evalEnv]["meanRet"]) * getRewardMultiplier(evalEnv, args.noRewardShaping)
        rewards.append(currentReward)
    print("Evaluate agent", rewards, "duration:", time.time() - startTime)
    return rewards
