from mpi4py import MPI
import numpy as np
import time

def populate_matrices(matrix_size):
    size = matrix_size // 2
    mat = np.random.randint(0, 11, size=(size, size))
    return mat

def c_submatrix_solver(c_m, a_1, b_1, a_2, b_2, temp):
    # Multiply
    np.dot(a_1, b_1, out=c_m)
    np.dot(a_2, b_2, out=temp)
    # Add
    np.add(c_m, temp, out=c_m)

def divide_n_conquer(c_mat, a_mat, b_mat):
    half = a_mat.shape[0]
    if half == 2:
        np.dot(a_mat, b_mat, out=c_mat)
        return
    a11, a12, a21, a22 = mat_splitter(a_mat)
    b11, b12, b21, b22 = mat_splitter(b_mat)
    c11, c12, c21, c22 = mat_splitter(c_mat)
    temp = np.zeros((a11.shape[0], a11.shape[0]))
    c_submatrix_solver(c11, a11, b11, a12, b21, temp)
    c_submatrix_solver(c12, a11, b12, a12, b22, temp)
    c_submatrix_solver(c21, a21, b11, a22, b21, temp)
    c_submatrix_solver(c22, a21, b12, a22, b22, temp)

def mat_splitter(mat):
    row = mat.shape[0]
    half = row // 2
    return mat[:half, :half], mat[:half, half:], mat[half:, :half], mat[half:, half:]

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    matrix_size = 8
    comp = populate_matrices(matrix_size)

    comm.Barrier()  # Synchronize all processes

    # Gather matrices to process 0
    all_matrices = comm.gather(comp, root=0)

    if rank == 0:
        # Perform multiplication
        t1 = time.time()
        res = np.zeros((matrix_size // 2, matrix_size // 2))
        divide_n_conquer(res, all_matrices[0], all_matrices[4])
        divide_n_conquer(res, all_matrices[1], all_matrices[5])
        divide_n_conquer(res, all_matrices[2], all_matrices[6])
        divide_n_conquer(res, all_matrices[3], all_matrices[7])
        t2 = time.time()
        print(f"time elapsed: {t2-t1}")
        print("Result Matrix:")
        print(res)
