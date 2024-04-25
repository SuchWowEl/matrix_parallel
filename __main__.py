import numpy as np
from mpi4py import MPI

def strassen_matrices_mult(A, B):
    n = A.shape[0]

    # Base case when size of matrices is 1x1
    if n == 1:
        return A * B

    # Divide matrices into quarters
    mid = n // 2
    A11, A12, A21, A22 = A[:mid, :mid], A[:mid, mid:], A[mid:, :mid], A[mid:, mid:]
    B11, B12, B21, B22 = B[:mid, :mid], B[:mid, mid:], B[mid:, mid:], B[mid:, mid:]

    # Recursive calls
    P1 = strassen_matrices_mult(A11 + A22, B11 + B22)
    P2 = strassen_matrices_mult(A21 + A22, B11)
    P3 = strassen_matrices_mult(A11, B12 - B22)
    P4 = strassen_matrices_mult(A22, B21 - B11)
    P5 = strassen_matrices_mult(A11 + A12, B11 + B12)
    P6 = strassen_matrices_mult(A21 - A11, B11 + B12)
    P7 = strassen_matrices_mult(A12 - A22, B21 + B22)

    # Combine P1-P7 into the final result
    C11 = P5 + P4 - P2 + P6
    C12 = P3 + P5
    C21 = P2 + P4
    C22 = P1 - P3 - P4 + P7

    # Construct the final matrix
    C = np.vstack((np.hstack((C11, C12)), np.hstack((C21, C22))))

    return C

def parallel_strassen_matrices_mult(A, B, comm):
    n = A.shape[0]
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Divide matrices into quarters
    mid = n // 2
    A11, A12, A21, A22 = A[:mid, :mid], A[:mid, mid:], A[mid:, :mid], A[mid:, mid:]
    B11, B12, B21, B22 = B[:mid, :mid], B[:mid, mid:], B[mid:, mid:], B[mid:, mid:]

    # Scatter matrices to all processes
    A11_loc, A12_loc, A21_loc, A22_loc = comm.scatter(A11), comm.scatter(A12), comm.scatter(A21), comm.scatter(A22)
    B11_loc, B12_loc, B21_loc, B22_loc = comm.scatter(B11), comm.scatter(B12), comm.scatter(B21), comm.scatter(B22)

    # Compute local parts of P1-P7
    P1_loc = strassen_matrices_mult(A11_loc + A22_loc, B11_loc + B22_loc)
    P2_loc = strassen_matrices_mult(A21_loc + A22_loc, B11_loc)
    P3_loc = strassen_matrices_mult(A11_loc, B12_loc - B22_loc)
    P4_loc = strassen_matrices_mult(A22_loc, B21_loc - B11_loc)
    P5_loc = strassen_matrices_mult(A11_loc + A12_loc, B11_loc + B12_loc)
    P6_loc = strassen_matrices_mult(A21_loc - A11_loc, B11_loc + B12_loc)
    P7_loc = strassen_matrices_mult(A12_loc - A22_loc, B21_loc + B22_loc)

    # Gather results from all processes
    P1 = comm.gather(P1_loc)
    P2 = comm.gather(P2_loc)
    P3 = comm.gather(P3_loc)
    P4 = comm.gather(P4_loc)
    P5 = comm.gather(P5_loc)
    P6 = comm.gather(P6_loc)
    P7 = comm.gather(P7_loc)

    # Combine P1-P7 into the final result
    C11 = P5 + P4 - P2 + P6
    C12 = P3 + P5
    C21 = P2 + P4
    C22 = P1 - P3 - P4 + P7

    # Construct the final matrix
    C = np.vstack((np.hstack((C11, C12)), np.hstack((C21, C22))))

    return C

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Initialize matrices
    n = 8192
    A = np.random.randint(0, 10, (n, n))
    B = np.random.randint(0, 10, (n, n))

    start_time = time.time()
    result = parallel_strassen_matrices_mult(A, B, comm)
    end_time = time.time()

    print(f"Execution time: {end_time - start_time:.2f} seconds")