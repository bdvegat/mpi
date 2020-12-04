from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

def send(rank,data,dest1):
    comm.send(data, dest=dest1, tag=11)

if rank == 0:
    data = [34,56,76,78,89,9]
    send(1,data,1)
    comm.send(data, dest=2, tag=11)
    comm.send(data, dest=3, tag=11)
else:
    data = comm.recv(source=0, tag=11)
    data=[x*rank for x in data]
    print(rank, " ",data)