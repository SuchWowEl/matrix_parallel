from mpi4py import MPI
import numpy as np
from numpy.lib import source
from numpy.matrixlib import matrix
import time

def populate_matrices(matrix_size, rank):
    size = matrix_size // 2
    
    # Populate matrix A
    mat = np.random.randint(0, 11, size=(size, size))

    print(f"This is process {rank}, my mat is: {mat}")

    return mat

def c_submatrix_solver(c_m, a_1, b_1, a_2, b_2, temp):
    # Multiply
    divide_n_conquer(c_m, a_1, b_1)
    divide_n_conquer(temp, a_2, b_2)

    # Add
    for x in range(a_1.shape[0]):
        for y in range(a_1.shape[0]):
            c_m[x,y] += temp[x,y]

def naive(c_mat, a_mat, b_mat):
    n = a_mat.shape[0]
    for i in range(n):
        for j in range(n):
            # c_mat[i][j] = 0
            for k in range(n):
                # if rank == 1:
                #     print(f"i: {i}, j: {j}, k: {k}")
                c_mat[i][j] += a_mat[i][k] * b_mat[k][j]

def divide_n_conquer(c_mat, a_mat, b_mat):
    # if rank == 1:
    #     print(f"dnc c_mat: {c_mat}")
    half = a_mat.shape[0]
    if half == 2:
        # if rank == 1:
        #     print(f"{rank} a_mat b4 naive: {a_mat}")
        #     print(f"{rank} b_mat b4 naive: {b_mat}")
        #     print(f"{rank} c_mat b4 naive: {c_mat}")
        naive(c_mat, a_mat, b_mat)
        return
    a11, a12, a21, a22 = mat_splitter(a_mat)
    b11, b12, b21, b22 = mat_splitter(b_mat)
    c11, c12, c21, c22 = mat_splitter(c_mat)
    # if rank == 1:
    #     print(f"dnc c11: {c11}, c12: {c12}, c21: {c21}, c22: {c22}")

    temp = np.zeros((a11.shape[0], a11.shape[0]))
    c_submatrix_solver(c11, a11, b11, a12, b21, temp)
    c_submatrix_solver(c12, a11, b12, a12, b22, temp)
    c_submatrix_solver(c21, a21, b11, a22, b21, temp)
    c_submatrix_solver(c22, a21, b12, a22, b22, temp)

def mat_splitter(mat):
    row = mat.shape[0]
    half = row//2
    return mat[:half, :half], mat[:half, half:], mat[half:, :half], mat[half:, half:]


# Example usage:
if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    matrix_size = 512
    comp = []

    tb = 0
    if rank == 0:
        tb = time.time()
    comp = populate_matrices(matrix_size, rank)

    comm.Barrier()  # Synchronize all processes

    # Gather matrices to process 0
    all_matrices = comm.gather(comp, root=0)
    if rank == 0:
        ta = time.time()
        print(f"time for number generation: {ta-tb}")

    # Print the matrices
    if rank == 0:
        print("BEFORE !!!!!!!!!!!!!!!!!!")
        print(f"M1.1: [ {all_matrices[0]} , {all_matrices[1]} ")
        print()
        print(f"M1.2: [ {all_matrices[2]} , {all_matrices[3]} ")
        print()
        print()
        print(f"M2.1: [ {all_matrices[4]} , {all_matrices[5]} ")
        print()
        print(f"M2.2: [ {all_matrices[6]} , {all_matrices[7]} ")
        # a11, a12, a21, a22 = all_matrices[0]

        # return mat[:half, :half], mat[:half, half:], mat[half:, :half], mat[half:, half:]
        t1 = time.time()
        half = all_matrices[0].shape[0]
        comm.send([all_matrices[0], all_matrices[4], all_matrices[1], all_matrices[6]], 1, 1)
        comm.send([all_matrices[0], all_matrices[5], all_matrices[1], all_matrices[7]], 2, 2)
        comm.send([all_matrices[2], all_matrices[4], all_matrices[3], all_matrices[6]], 3, 3)
        comm.send([all_matrices[2], all_matrices[5], all_matrices[3], all_matrices[7]], 4, 4)
        res1 = comm.recv(source=1,tag=6)
        res2 = comm.recv(source=2,tag=7)
        res3 = comm.recv(source=3,tag=8)
        res4 = comm.recv(source=4,tag=9)
        t2 = time.time()

        print(f"time elapsed: {t2-t1}")

        print("AFTER !!!!!!!!!!!!!!!!!!")
        print(f"M1.1: [ {res1} , {res2} ")
        print()
        print(f"M1.2: [ {res3} , {res4} ")
    elif rank >= 1 and rank <= 4:
        workloads = comm.recv(source=0,tag=rank)
        # matrix_size = matrix_size // 2
        temp = np.zeros([workloads[1].shape[0], workloads[1].shape[0]]) #[[0]*workloads[0].shape[0]]*workloads[0].shape[0] 
        c_sub = np.zeros([workloads[1].shape[0], workloads[1].shape[0]])
        # print(f"temp: {temp}")
        c_submatrix_solver(c_sub, workloads[0], workloads[1], workloads[2], workloads[3], temp)
        print(f"temp outside: {c_sub}")
        comm.send(c_sub,0,rank+5)
