import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

difficultyStepsize = 100000
iterationsDone = np.linspace(0, 5000000, 1000)


def calculateMaxSteps(x):
    startDecreaseNum = 500000
    if x <= startDecreaseNum:
        value: float = 1.0
    else:
        value = 1 - ((x - startDecreaseNum) / difficultyStepsize / 20)
    value = max(value, 0.15)
    return value


# Create a vectorized version of the function to apply on a numpy array
vcalc_value = np.vectorize(calculateMaxSteps)

y_values = vcalc_value(iterationsDone)
sns.set(style="darkgrid")

plt.figure(figsize=(10, 6))
sns.lineplot(x=iterationsDone, y=y_values)
plt.xlabel('training iterations done')
plt.ylabel('maximum steps %')
# plt.title('')
plt.ylim(0, 1.01)
plt.xlim(0, 5000000)
plt.show()
