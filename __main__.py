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
    A[row_start:row_end, :] = np.random.randint(10, size=(submatrix_size, matrix_size), dtype=np.int32)

    # Populate matrix B
    B[:, row_start:row_end] = np.random.randint(10, size=(matrix_size, submatrix_size), dtype=np.int32)

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

    # Broadcast the shape of matrices to all processes
    A_shape = np.empty(2, dtype=np.int32)
    B_shape = np.empty(2, dtype=np.int32)
    comm.Bcast([A_shape, MPI.INT32_T], root=0)
    comm.Bcast([B_shape, MPI.INT32_T], root=0)

    if A_shape[1] != B_shape[0]:
        raise ValueError("Matrices must be of compatible dimensions for multiplication")

    # Print the shapes of A and B after broadcasting
    print(f"Process {rank}: Shape of A: {A_shape}, Shape of B: {B_shape}")

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

    if rank == 0:
        matrix_size = 4
        A = np.empty((matrix_size, matrix_size), dtype=np.int32)
        B = np.empty((matrix_size, matrix_size), dtype=np.int32)
    else:
        A = None
        B = None

    result = matrix_multiply_divide_and_conquer(A, B, comm)
