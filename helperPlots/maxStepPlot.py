import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

iterationSteps = 75000
totalIterations = 5000000
startDecreasingNum = 500000
sns.set(style="darkgrid")
sns.set(font_scale=1.3)
plt.figure(figsize=(10, 6))


def calculateMaxSteps(iterDone, difficultyStepSize=2000000, startDecreaseNum=500000):
    if iterDone <= startDecreaseNum:
        value: float = 1.0
    else:
        value = 1 - ((iterDone - startDecreaseNum) / difficultyStepSize)
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
plt.ylim(0, 1.005)
plt.xlim(0, totalIterations)
plt.show()
