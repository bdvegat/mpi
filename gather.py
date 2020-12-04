from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

data = [34,56,76,78,89,9]*rank
dat = comm.gather(data, root=0)
if rank == 0:
    for i in range(size):
        print(dat[i])