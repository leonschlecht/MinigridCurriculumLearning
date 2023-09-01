from scipy.stats import qmc
import matplotlib.pyplot as plt

from utils.curriculumHelper import *

l_bounds = [0.2, 0.2]
u_bounds = [0.8, 0.8]
sampler = qmc.Sobol(d=2, seed=42)
samples = sampler.random_base2(m=3)
samples = qmc.scale(samples, l_bounds, u_bounds)

finalSamples = []
for c, m in samples:
    finalSamples.append([round(c, 2), round(m, 2)])

points = finalSamples

x = [point[0] for point in points]
y = [point[1] for point in points]

# adjust sample to match existing experiment
x[-1] = .8
y[-1] = .8

plt.scatter(x, y)
plt.xlabel('Mutation Rate', fontsize=labelFontsize)
plt.ylabel('Crossover Rate', fontsize=labelFontsize)
plt.title('Varying of Crossover and Mutation Rates', fontsize=titleFontsize)
print(finalSamples)
plt.show()

