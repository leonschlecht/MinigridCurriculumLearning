import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

iterationSteps = 100000
totalIterations = 5000000
startDecreasingNum = 500000
sns.set(style="darkgrid")
plt.figure(figsize=(10, 6))


def calculateMaxSteps(iterDone, smoothingFactor=20, difficultyStepSize=100000, startDecreaseNum=500000):
    if iterDone <= startDecreaseNum:
        value: float = 1.0
    else:
        value = 1 - ((iterDone - startDecreaseNum) / difficultyStepSize / smoothingFactor)
    value = max(value, 0.15)
    return value


x = np.arange(0, totalIterations)
y = np.ones_like(x, dtype=float)
for i in range(startDecreasingNum - iterationSteps, totalIterations, iterationSteps):
    y[i: i + iterationSteps] = calculateMaxSteps(i)
data = pd.DataFrame({'x': x, 'y': y})
ax = sns.lineplot(data=data, x='x', y='y')

ax.set_xlabel('Training Iterations Done', fontsize='x-large')
ax.set_ylabel('Maximum Steps %', fontsize='x-large')
plt.title("Visualization of Maximum Steps Decreasing", fontsize="x-large")

ax.xaxis.set_tick_params(labelsize='medium')
ax.yaxis.set_tick_params(labelsize='medium')
plt.ylim(0, 1.01)
plt.xlim(0, totalIterations)
plt.show()
