import matplotlib.pyplot as plt
import numpy as np
import math

a1 = 3
a2 = 2
b1 = -1
b2 = 1
c1 = 2
c2 = 3.2391

def p(x):
    return math.cos(x)

def q(x):
    return math.sin(x)

def f(x):
    return math.sin(x) * x

def ExactSolution(x):
    return x + math.cos(x)

def SweepMethod(min, max, step, flag):
    massX = np.arange(min, max+step/2, step)
    massP = []
    massQ = []
    massF = []
    n = len(massX)

    for x in massX:
        massP.append(p(x))
        massQ.append(q(x))
        massF.append(f(x))

    if flag == 0:
        massA = [0]
        massB = [a1-b1/step]
        massC = [b1/step]
        massD = [c1]

    if flag != 0:
        massA = [0]
        massB = [-2 + 2*step*a1/b1 + a1*step*step*massP[0]/b1 + massQ[0]*step*step]
        massC = [2]
        massD = [massF[0]*step*step + c1*2*step/b1 - step*step*c1/b1]

    for i in range(1, n-1):
        massA.append(1 - step*massP[i]/2)
        massB.append(-2 + step*step*massQ[i])
        massC.append(1 + massP[i]/2*step)
        massD.append(massF[i]*step*step)

    if flag == 0:
        massA.append(-b2/step)
        massB.append(a2 + b2 / step)
        massC.append(0)
        massD.append(c2)

    if flag != 0:
        massA.append(2)
        massB.append(-2 - 2*step*a2/b2 - massP[-1]*step*step*a2/b2 + massQ[-1]*step*step)
        massC.append(0)
        massD.append(massF[-1]*step*step + step*step*massP[-1]*c2/b2 - 2*step*c2/b2)

    massA3 = [-massC[0] / massB[0]]
    massB3 = [massD[0] / massB[0]]

    for i in range(1, n):
        massA3.append(-massC[i] / (massB[i] + massA[i]*massA3[i-1]))
        massB3.append((massD[i] - massA[i] * massB3[i-1]) / ((massB[i] + massA[i]*massA3[i-1])))

    massResultY = [massB3[-1]]
    for i in range(n - 2, -1, -1):
        massResultY.append(massB3[i] + massA3[i]*massResultY[len(massX) - 2 - i])

    massX = np.arange(min, max+step/2, step)
    print(massResultY[::-1])
    return massX, massResultY[::-1]

def LogError(min, maximum, Func):
    step = 0.1
    massErr = []
    massStep = []
    while(step > 1e-4):
        massCalcX, massCalcY = Func(min, maximum, step)
        massTrue = [ExactSolution(i) for i in np.arange(min, max+step/2, step)]

        print(len(massTrue))
        print(len(massCalcY))
        maxErr = 0
        for i in range(len(massTrue)):
            if maxErr < abs(massCalcY[i] - massTrue[i]):
                maxErr = abs(massCalcY[i] - massTrue[i])

        massStep.append(math.log(step))
        massErr.append(math.log(maxErr))
        step = step / 10

    return massStep, massErr

step = 0.05
min = 0
max = 1
massX, massY = SweepMethod(min, max, step, 1)
# massStep, massErr = LogError(min, max, SweepMethod)
plt.title("19 задание")  # заголовок/
plt.xlabel("x")
plt.ylabel("y")
plt.grid()
plt.plot(massX, massY, label='решение по методу прогонки')
plt.plot(massX, [ExactSolution(i) for i in massX], label='истинное решение')
# plt.plot(massStep, massErr, label='функция логарифма ошибки')
plt.legend()
plt.show()
