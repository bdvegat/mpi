import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import axes3d



neurons = [10,50,100,150,200]
batch = [100,200,300,400,500]

time_str = "time"
batch_size = "batch size"

# 10 neuronas
secuencial = [2.1513, 3.4146, 4.7030, 6.1434, 7.6210]
cuda = [11.3847, 9.8396, 9.9263, 10.1628, 10.1492]
time = [secuencial, cuda]
plt.ylabel(time_str)
plt.xlabel(batch_size)
plt.plot(batch,time)
plt.savefig("10n.svg")
plt.show()
#50 neuronas
secuencial = [8.1204, 22.5635, 33.3338, 42.0032, 51.7528]
cuda = [10.2272, 10.2673, 9.9345, 9.9485, 9.9807]
time = [secuencial, cuda]
plt.ylabel(time_str)
plt.xlabel(batch_size)
plt.savefig("50n.svg")
plt.show()
#100 neuronas
secuencial = [25.5575, 47.0661, 68.1980, 95.2925, 116.9705]
cuda = [10.1672, 10.0597, 10.0083, 9.9588, 10.1131]
time = [secuencial, cuda]
plt.ylabel(time_str)
plt.xlabel(batch_size)
plt.savefig("100n.svg")
plt.show()
#150 neuronas
secuencial = [39.8597, 70.5275, 109.5580, 139.9209, 178.5339]
cuda = [9.9712, 9.9535, 10.1527, 9.9302, 11.4195]
time = [secuencial, cuda]
plt.ylabel(time_str)
plt.xlabel(batch_size)
plt.savefig("150n.svg")
plt.show()
#200 neuronas
secuencial = [58.2922, 110.7287, 155.2711, 202.7360, 252.7381]
cuda = [ 9.8133, 9.9344, 11.3006, 14.5094, 17.7260]
time = [secuencial, cuda]
plt.ylabel(time_str)
plt.xlabel(batch_size)
plt.savefig("200n.svg")
plt.show()

#250 neuronas
secuencial = [75.8357, 140.4651, 195.5719, 264.2548, 326.2062]
cuda = [9.8555, 11.0743, 15.1743, 19.5431, 24.7379]
time = [secuencial, cuda]
plt.ylabel(time_str)
plt.xlabel(batch_size)
plt.savefig("200n.svg")
plt.show()