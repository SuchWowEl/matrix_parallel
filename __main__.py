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

    # Populate matrices A and B in parallel
    populate_matrices(A, B, comm)

    # Divide work along the rows of matrices
    rows_per_process = A_shape[0] // size
    start_row = rank * rows_per_process
    end_row = start_row + rows_per_process if rank != size - 1 else A_shape[0]

    # Scatter matrix A among processes
    local_A = np.empty((end_row - start_row, A_shape[1]), dtype=np.int32)
    comm.Scatter([A, MPI.INT32_T], [local_A, MPI.INT32_T], root=0)

    # Broadcast matrix B to all processes
    comm.Bcast([B, MPI.INT32_T], root=0)

    # Base case: if the submatrices are small enough, perform regular matrix multiplication
    if A_shape[0] <= 128 or A_shape[1] <= 128 or B_shape[1] <= 128:
        local_C = np.dot(local_A, B)
    else:
        # Divide matrices into submatrices
        mid = A_shape[1] // 2

        A11 = local_A[:, :mid]
        A12 = local_A[:, mid:]
        B11 = B[:mid]
        B12 = B[mid:]

        # Send and receive submatrices
        A_send = np.empty_like(A11)
        B_recv = np.empty_like(B11)

        comm.Sendrecv(A12, dest=(rank + 1) % size, sendtag=1,
                      recvbuf=A_send, source=(rank - 1 + size) % size, recvtag=1)
        comm.Sendrecv(B12, dest=(rank + 1) % size, sendtag=2,
                      recvbuf=B_recv, source=(rank - 1 + size) % size, recvtag=2)

        # Perform local matrix multiplication recursively
        C11 = matrix_multiply_divide_and_conquer(A11, B11, comm)
        C12 = matrix_multiply_divide_and_conquer(A_send, B_recv, comm)

        # Reuse previously declared matrix for local_C
        local_C = np.empty((C11.shape[0], C11.shape[1] + C12.shape[1]), dtype=np.int32)
        local_C[:, :C11.shape[1]] = C11
        local_C[:, C11.shape[1]:] = C12

    # Gather results from all processes
    if rank == 0:
        result = np.empty((A_shape[0], B_shape[1]), dtype=np.int32)
    else:
        result = None
    comm.Gather([local_C, MPI.INT32_T], [result, MPI.INT32_T], root=0)

    return result

# Example usage:
if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
        matrix_size = 65536
        A = np.empty((matrix_size, matrix_size), dtype=np.int32)
        B = np.empty((matrix_size, matrix_size), dtype=np.int32)
    else:
        A = None
        B = None

    result = matrix_multiply_divide_and_conquer(A, B, comm)
    if rank == 0:
        print(result)
