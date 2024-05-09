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
    comp = []

    comp = populate_matrices(matrix_size, rank)

    comm.Barrier()  # Synchronize all processes

    # Gather matrices to process 0
    all_matrices = comm.gather(comp, root=0)

    # Print the matrices
    if rank == 0:
        print(f"M1.1: [ {all_matrices[0]} , {all_matrices[1]} ")
        print()
        print(f"M1.2: [ {all_matrices[2]} , {all_matrices[3]} ")
        print()
        print()
        print(f"M2.1: [ {all_matrices[4]} , {all_matrices[5]} ")
        print()
        print(f"M2.2: [ {all_matrices[6]} , {all_matrices[7]} ")