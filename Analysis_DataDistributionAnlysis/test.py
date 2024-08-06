from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as np


def func(x, a, b, c):
    return a * np.exp(-b * x) + c



a = [0,3,4,5]
b = [i for i, e in enumerate(a) if e != 0]
print(b)
c = [a[i] for i in b]
print(c)






xdata = np.linspace(0, 4, 50)
y = func(xdata, 2.5, 1.3, 0.5)
#ydata = y + 200 * np.random.normal(size=len(xdata))
ydata = y + 200

plt.plot(xdata, ydata, 'b-')
popt, pcov = curve_fit(func, xdata, ydata)
# popt数组中，三个值分别是待求参数a,b,c
y2 = [func(i, popt[0], popt[1], popt[2]) for i in xdata]
plt.plot(xdata, y2, 'r--')
plt.show()