from mpi4py import MPI
import numpy as np

def populate_matrices(A, B, comm):
    """
    Parallel population of matrices A and B.
    
    Args:
    A: First matrix (2D numpy array)
    B: Second matrix (2D numpy array)
    comm: MPI communicator
    """
    rank = comm.Get_rank()
    size = comm.Get_size()

    matrix_size = A.shape[0]
    submatrix_size = matrix_size // size  # Divide the matrix size by the number of processes

    # Define the ranges for rows and columns for each process
    row_start = rank * submatrix_size
    row_end = row_start + submatrix_size

    # Populate matrix A
    A[row_start:row_end, :] = np.random.randint(0, 11, size=(submatrix_size, matrix_size))

    # Populate matrix B
    B[row_start:row_end, :] = np.random.randint(0, 11, size=(submatrix_size, matrix_size))

def matrix_multiply_divide_and_conquer(A, B, comm):
    """
    Parallel matrix multiplication using MPI with divide and conquer method.
    
    Args:
    A: First matrix (2D numpy array)
    B: Second matrix (2D numpy array)
    comm: MPI communicator
    
    Returns:
    Result of the matrix multiplication (2D numpy array)
    """
    rank = comm.Get_rank()
    size = comm.Get_size()
    matrix_size = 4

    # Initialize the shape information in the root process
    if rank == 0:
        matrix_size = 4
        A = np.zeros((matrix_size, matrix_size))
        B = np.zeros((matrix_size, matrix_size))
        shape = np.array(A.shape)
    else:
        shape = np.empty(2, dtype=np.int32)

    # Broadcast the shape of matrices to all processes
    comm.Bcast([shape, MPI.INT32_T], root=0)

    print(f"Process {rank}: Shape of A and B: {shape}")  # Print shapes of A and B

    if shape[1] != shape[0]:
        raise ValueError("Matrices must be of compatible dimensions for multiplication")

    # Populate matrices A and B in parallel
    populate_matrices(A, B, comm)

    # Print the populated matrices A and B
    print(f"Process {rank}: Matrix A:\n{A}")
    print(f"Process {rank}: Matrix B:\n{B}")

    return


# Example usage:
if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    m1 = m2 = []

    result = matrix_multiply_divide_and_conquer(m1, m2, comm)
