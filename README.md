# RHEA CL (MiniGrid)
This is the repository of my master thesis. 
We conducted research of using the combination of Curriculum Learning (CL) with Rolling Horizon Evolutionary Algorithms (RHEA).
All the logs  (in /storage) and scripts (in ./) are contained in this repository.
The PPO implementation and rough overview of this repository is taken from https://github.com/lcswillems/rl-starter-files
----

### Training an agent:
- install requirements: pip3 install requirements.txt -r
Training with RHEA CL:
- run script: python3 trainCurriculum.py --procs 24 --stepsPerCurric 3 --nGen 4 --numCurric 2 --iterPerEnv 100000 --model 150k_3step_4gen_2curric --noRewardShaping --seed 1 
- See full list of parameters with --help
- If you want to change the environments, you need to do so in the ENV_NAMES.py file, and then update the trainCurriculum.py file so that you select this environment

----

### Evaluation script:
Example for showing the Performance plot, grouped by iterationSteps
- python3 ./evaluateCurriculum.py --iterGroup --rhea --scores curric
- Need to either specifiy --scores, --trainingTime, --curricDistr or --splitDistr
- Allows for filtering of 1 hyperparameter, or comparison of n models. Start the script with --comparisons
- There are filter Options like --rhea, --ga, --nsga, --env etc available as well. Unfortunately, not everything will work 100% as expected, as some quickfixes were implemented to make specific plots work
---
### Visualization
You can visualize the performance of an agent.
For example, in the 5x5 MiniGrid-DoorKey:
python3 ./visualize.py --model <modelName> --env MiniGrid-DoorKey-8x8-v0

--------

### Some further notes:
- If you cancel an experiment, which you want to continue, it will start at the beginning of the last epoch. This might lead to having to re-do 1 or 2 generations. You need to make sure that the args parameters are still correctly set.
- all logs and model files are stored in /storage
- Can train RHEA CL, Random RH, AllParallel (training all envs at simulatneously), SPCL
- You can also look at tensorboard to see the reward during training with this command: tensorboard --logdir path\To\Repo\storage
- The sobol sampling points and the max step reduction plots are in the ./helperPlots directory
