from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

sendbuf = np.array([[1],[2,2],[3,3,3],[4,4,4,4]])
recvbuf = None
if rank == 0:
    recvbuf = np.vstack((sendbuf,sendbuf,sendbuf,sendbuf))
    print(recvbuf)
comm.Gatherv(sendbuf=sendbuf, recvbuf=(recvbuf,16), root=0)

if rank == 0:
    for i in range(0,len(recvbuf)):
        for j in range(0, len(recvbuf[i])):
            print(recvbuf[i,j],end=" ")
        print()
        #assert np.allclose(recvbuf[i,:], i)