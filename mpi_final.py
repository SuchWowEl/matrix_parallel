from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

np.random.seed(4)

if rank == 0:
    # initialize the matrix
    matrix = np.random.randint(low=0, high=10, size=(23, 4))
    row = matrix.shape[0]
    sendbuf = np.arange(row)
    print(sendbuf)

    # count: the size of each sub-task
    ave, res = divmod(sendbuf.size, size)
    count = [ave + 1 if p < res else ave for p in range(size)]
    # count = []
    # for p in range(size):
    #     if p < res:
    #         count.append(ave+1)
    #     else:
    #         count.append(ave)
    count = np.array(count)
    

    # displacement: the starting index of each sub-task
    displ = [sum(count[:p]) for p in range(size)]
    # displ = [sum(count[:p]) for p in range(size)]
    displ = np.array(displ)
else:
    sendbuf = None
    # initialize count on worker processes
    count = np.zeros(size, dtype=int)
    displ = np.zeros(size, dtype=int)
    matrix = np.random.randint(low=0, high=10, size=(23, 4))
    matrix = np.zeros(shape=(5,4), dtype=int)

# broadcast count
comm.Bcast(count, root=0)
comm.Bcast(displ, root=0)

# initialize recvbuf on all processes
recvbuf = np.array(list((0 for i in range(count[rank]))))

comm.Scatterv([sendbuf, count, displ, MPI.INT], recvbuf, root=0)


a = comm.bcast(matrix, root=0)
# print("Process", rank, a)
matrix_slice = []
for index in recvbuf:
    matrix_slice.append(a[index])
matrix_slice = np.array(matrix_slice, dtype=int)

matrix_2 = np.random.randint(low=0, high=10, size=(4, 5))

matrix_slice_buf = np.zeros(shape=(count[rank], matrix_2.shape[1]), dtype=int)

# first matrix is matrix_slice
# 2nd matrix is matrix_2

row_count = 0
column_count = 0
for row in matrix_slice:
    column_count = 0
    for col in matrix_2.T:
        matrix_slice_buf[row_count][column_count] = sum(row*col)
        column_count += 1
    row_count += 1

print("Process", rank, matrix_slice)
matrix_slice_buf = np.array([x for x in (y for y in matrix_slice_buf)])


check = False

if rank == 0:
    final_matrix = np.zeros(shape=(a.shape[0], matrix_2.shape[1]), dtype=int)
    check = True
else:
    final_matrix = None

sendcounts = np.array(comm.gather(matrix_slice_buf.size, 0))


comm.Gatherv(matrix_slice_buf, (final_matrix, sendcounts), root=0)

if rank == 0:
    print("Here is the matrix multiplication result using MPI:\n")
    print(final_matrix)
    
    print("\n\nMatrix A\n", a)
    print("\nMatrix B\n", matrix_2)

    print("Here is the result using np.dot()")
    print(np.dot(a, matrix_2))
# if comm.rank == 0:
#     print("AHHA")
#     print(final_matrix)
#     MPI.Finalize()

# if comm.rank == 0:
#     print("RANK 0")

#     for i in range(size):
#         print("rank is", rank)
#         matrix_slice_buf = comm.recv(matrix_slice_buf, source=rank)
#         print(matrix_slice_buf)
#         final_matrix[displ[rank]:matrix_slice_buf.shape[0]] += matrix_slice_buf

#     print(final_matrix)

