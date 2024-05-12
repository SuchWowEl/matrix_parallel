from mpi4py import MPI
import numpy as np
import time

def populate_matrices(matrix_size, rank):
    size = matrix_size // 2
    
    # Populate matrix A
    mat = np.random.randint(0, 11, size=(size, size))

    print(f"This is process {rank}, my mat is: {mat}")

    return mat

def c_submatrix_solver(c_m, a_1, b_1, a_2, b_2, temp):
    # Multiply
    np.matmul(a_1, b_1, out=temp)
    np.matmul(a_2, b_2, out=c_m)

    print("After multiplication:")
    print(c_m)

    # Add
    c_m += temp

    print("After addition:")
    print(c_m)


def distribute_workload(comm, matrix_size):
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        tb = time.time()

    # Generate and scatter matrices
    mat_size = matrix_size // 2
    comp = populate_matrices(matrix_size, rank)
    local_comp = np.zeros((mat_size, mat_size))
    comm.Scatter(comp, local_comp, root=0)

    comm.Barrier()  # Synchronize all processes

    print(f"This is process {rank}, my local_comp is: {local_comp}")
    print(f"This is process {rank}, while my comp is: {comp}")

    # Perform matrix multiplication
    temp = np.zeros((mat_size, mat_size))
    c_sub = np.zeros((mat_size, mat_size))
    c_submatrix_solver(c_sub, local_comp[0], local_comp[1], local_comp[2], local_comp[3], temp)

    # Gather results
    all_results = comm.gather(c_sub, root=0)

    if rank == 0:
        ta = time.time()
        print(f"time for number generation: {ta - tb}")
        print("Results:", all_results)

if __name__ == "__main__":
    comm = MPI.COMM_WORLD

    matrix_size = 512
    distribute_workload(comm, matrix_size)
