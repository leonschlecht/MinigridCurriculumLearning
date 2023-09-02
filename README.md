# MinigridCurriculumLearning
- Manually change the algorithm in the code. No mention of the algorithm in the final logs
- doesnt delete previous epoch models. Much is duplicated and redundant / bloats up the storage folder a bit with tfevents etc
- No seed reloading check. If want to continue, must ensure that the param --seed is set
- Same goes for iterPerEnv, numCurric, stepsPerCurric parameters. 
- all logs and model files are stored in /storage
- Can train RHEA, RRH, AllParalell, AllParallel as Curriculum, SPLCL or just a single env
- can visualize with scripts.evaluateCurriculum
- train a model with scripts.trainCurriuclum



TODO
- wie funktioniert das ganze: mit dem _CANDIDATE usw, wie dann weitergemacht wird. 
- Dass man reloaden kann und zu beginn der epoche weitermachen kann

-------------------

Evaluation:
Filter for rhea runs only:
python -m scripts.evaluateCurriculum --rhea 


### Crossover Mutation experiments
python3 -m scripts.evaluateCurriculum --normalize --curricDistr --errorbar se --crossoverMutation


xIterations: how many iterations to show on the x axis
errorbar: ci | sd | se | None (default)




If you want to change DoorKey Environments, you need to do so in the ENV_NAMES.py file