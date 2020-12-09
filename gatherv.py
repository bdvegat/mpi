from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size

w = np.array([[1.0,2.0,3.0,4.0,5.0],[6.0,7.0],[9.0,10.0,11.0,1.0],[0]])
print(w.shape)
arr = np.concatenate(w)*rank
print(arr.shape)
recvbuf = None
if rank == 0:
    recvbuf = np.empty([size,len(arr)],dtype = 'd')
comm.Gather(arr, recvbuf,root=0)
if rank == 0:
    recvbuf = recvbuf.sum(axis=0)
    print(recvbuf)
    k = 0
    for i in range (0,len(w)):
        for j in range(0,len(w[i])):
            print("w = {} + {}".format(w[i][j],recvbuf[k]))
            w[i][j] = w[i][j] + recvbuf[k]
            print(w[i][j])
            k+=1
    print(w)