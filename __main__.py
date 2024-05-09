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

    print(f"M1 is: {m1}")
    print(f"M2 is: {m2}")
