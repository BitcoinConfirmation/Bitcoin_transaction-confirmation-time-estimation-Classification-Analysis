import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0, 10, 1000)
y = np.sin(t)
plt.plot(t, y)
b=5
c=str(b)+'x'
plt.xlabel('y=a'+'$\mathregular{e^{cx}}$'+'+b', fontdict={'weight': 'normal', 'size': 15})
plt.show()


