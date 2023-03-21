import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax = fig.add_subplot(111)
x = [1, 3, 6, 9, 16, 25, 169, 676, 2704]
y1 = [0.194099665, 0.204020977, 0.203106046, 0.187581778, 0.185601115, 0.214554071, 0.215484381, 0.195391059, 0.206275225]
y2 = [0.436446667, 1.40184772, 1.5317130089, 1.865842462, 1.755299568, 1.84389925, 4.602034688, 16.3081187, 59.7960794]
y3 = [2.255722523, 7.089862823, 8.035542846, 11.06607938, 10.02075803, 11.12364435, 27.82798851, 115.9677161, 457.2613702]
ax.plot(x,y1, label="NoC")
ax.plot(x,y2, label="Elbow method")
ax.plot(x,y3, label="Gap statistic method")
ax.set_xlabel("log scale of number of cluster", fontsize = 12)
ax.set_ylabel("log scale of runtime, s", fontsize = 12)
ax.legend()
ax.grid()
ax.set_yscale('log')
ax.set_xscale('log')
plt.show()

print(np.mean(y1))
print(np.std(y1))