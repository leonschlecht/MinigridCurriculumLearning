# MinigridCurriculumLearning
 Current Version: (June)
- Manually change the algorithm in the code. No mention of the algorithm in the final logs
- doesnt delete previous epoch models. Much is duplicated and redundant / bloats up the storage folder a bit with tfevents etc
- No seed reloading check. If want to continue, must ensure that the param --seed is set
- Same goes for iterPerEnv, numCurric, stepsPerCurric parameters. 
- all logs and model files are stored in /storage
- Can train RHEA, RRH, AllParalell (baseline)
- can visualize with scripts.evaluateCurriculum
- train a model with scripts.trainCurriuclum

-------------------

Evaluation:
Filter for rhea runs only:
python -m scripts.evaluateCurriculum --rhea 


xIterations: how many iterations to show on the x axis
errorbar: ci | sd | None (default)