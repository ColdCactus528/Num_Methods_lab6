import matplotlib.pyplot as plt
import numpy as np
import math

a1 = 2
a2 = 3
b1 = -1
b2 = 1
c1 = -1
c2 = 8.064715260291829

def p(x):
    return math.tan(x)

def q(x):
    return -2*x / math.cos(x)

def f(x):
    return 2 - 2*x*x*x / math.cos(x)

def ExactSolution(x):
    return x*x + math.sin(x)

def SweepMethod(min, max, n, flag):
    massX = np.linspace(min, max, n)
    massP = []
    massQ = []
    massF = []
    step = (max - min) / (n-1)

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
        massB = [-2 + 2*step*a1/b1 - a1*step*step*massP[0]/b1 + massQ[0]*step*step]
        massC = [2]
        massD = [massF[0]*step*step - c1*step*step*massP[0]/b1 + 2*step*c1/b1]

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
        massD.append(massF[-1]*step*step - step*step*massP[-1]*c2/b2 - 2*step*c2/b2)

    massA3 = [-massC[0] / massB[0]]
    massB3 = [massD[0] / massB[0]]

    for i in range(1, n):
        massA3.append(-massC[i] / (massB[i] + massA[i]*massA3[i-1]))
        massB3.append((massD[i] - massA[i] * massB3[i-1]) / ((massB[i] + massA[i]*massA3[i-1])))

    massResultY = [massB3[-1]]
    for i in range(n - 2, -1, -1):
        massResultY.append(massB3[i] + massA3[i]*massResultY[n - 2 - i])

    return massX, massResultY[::-1]

def LogError(min, maximum, n, Func, flag):
    massErr = []
    massStep = []
    mass1 = []
    mass2 = []
    while(n < 1000):
        massCalcX, massCalcY = Func(min, maximum, n, flag)
        massTrue = [ExactSolution(i) for i in np.linspace(min, maximum, n)]

        print(massTrue)
        print(massCalcY)
        maxErr = 0
        for i in range(len(massTrue)):
            if maxErr < abs(massCalcY[i] - massTrue[i]):
                maxErr = abs(massCalcY[i] - massTrue[i])

        step = (maximum - min) / (n - 1)
        mass1.append(step)
        mass2.append(maxErr)
        massStep.append(math.log(step))
        massErr.append(math.log(maxErr))
        n = n * 2

    return massStep, massErr, mass1, mass2

step = 0.05
n = 21
min = 0
max = 1
massX, massY = SweepMethod(min, max, n, 0)
massStep, massErr, massStepNoLog, massErrNoLog = LogError(min, max, n, SweepMethod, 1)
print(massErrNoLog)
print()
print(massStepNoLog)
plt.title("19 задание")  # заголовок/
plt.xlabel("x")
plt.ylabel("y")
plt.grid()
plt.plot(massX, massY, label='решение по методу прогонки')
plt.plot(massX, [ExactSolution(i) for i in massX], label='истинное решение')
# plt.plot(massStep, massErr, label='функция логарифма ошибки')
plt.legend()
plt.show()
