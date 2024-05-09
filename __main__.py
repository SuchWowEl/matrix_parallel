from mpi4py import MPI
import numpy as np

def populate_matrices(matrix_size, rank):
    size = matrix_size // 4
    
    # Populate matrix A
    matrix = np.random.randint(0, 11, size=(size, size))

    print(f"This is process {rank}, my matrix is: {matrix}")

    return matrix


# Example usage:
if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    matrix_size = 16
    m1 = [[] for _ in range(4)]
    m2 = [[] for _ in range(4)]

    if rank < 4:
        m1[rank] = populate_matrices(matrix_size, rank)
    else:
        m2[rank-4] = populate_matrices(matrix_size, rank)

    comm.Barrier()  # Synchronize all processes

    # Gather matrices to process 0
    all_m1 = comm.gather(m1, root=0)
    all_m2 = comm.gather(m2, root=0)

    # Print the matrices
    if rank == 0:
        for i in range(2):
            print("M1:", all_m1[i])
            print("M2:", all_m2[i])
        print()
        for i in range(2, 4):
            print("M1:", all_m1[i])
            print("M2:", all_m2[i])