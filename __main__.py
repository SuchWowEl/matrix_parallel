from mpi4py import MPI
import numpy as np
import random

comm = MPI.COMM_WORLD
rank = comm.rank

def matrix_randomizer(n):
    matrix = np.random.randint(0, 11, size=(n, n))
    print(f"matrix: {matrix}")
    return matrix

def split(matrix):
    row, col = matrix.shape
    row2, col2 = row//2, col//2
    return matrix[:row2, :col2], matrix[:row2, col2:], matrix[row2:, :col2], matrix[row2:, col2:]

def mini_strassen(mat1, mat2):
    if len(mat1) == 1:
        return mat1 * mat2

    a, b, c, d = split(mat1)
    e, f, g, h = split(mat2)

    p1 = mini_strassen(a, f - h) 
    p2 = mini_strassen(a + b, h)     
    p3 = mini_strassen(c + d, e)     
    p4 = mini_strassen(d, g - e)     
    p5 = mini_strassen(a + d, e + h)     
    p6 = mini_strassen(b - d, g + h) 
    p7 = mini_strassen(a - c, e + f) 

    c11 = p5 + p4 - p2 + p6 
    c12 = p1 + p2         
    c21 = p3 + p4         
    c22 = p1 + p5 - p3 - p7 

    c = np.vstack((np.hstack((c11, c12)), np.hstack((c21, c22)))) 

    return c

def strassen(n, m1, m2, rank):
    if len(m1) == 1:
        return m1 * m2

    a, b, c, d = split(m1)
    e, f, g, h = split(m2)

    p1 = p2 = p3 = p4 = p5 = p6 = p7 = []

    if (rank==0):
        p1 = comm.recv(source=1,tag=1)
        p2 = comm.recv(source=2,tag=2)
        p3 = comm.recv(source=3,tag=3)
        p4 = comm.recv(source=4,tag=4)
        p5 = comm.recv(source=5,tag=5)
        p6 = comm.recv(source=6,tag=6)
        p7 = comm.recv(source=7,tag=7)

    if (rank==1):
        p1 = mini_strassen(a, f - h)
        comm.send(p1,0,rank)
    if (rank==2):
        p2 = mini_strassen(a + b, h)
        comm.send(p2,0,rank)
    if (rank==3):
        p3 = mini_strassen(c + d, e)
        comm.send(p3,0,rank)
    if (rank==4):
        p4 = mini_strassen(d, g - e)
        comm.send(p4,0,rank)
    if (rank==5):
        p5 = mini_strassen(a + d, e + h)
        comm.send(p5,0,rank)
    if (rank==6):
        p6 = mini_strassen(b - d, g + h)
        comm.send(p6,0,rank)
    if (rank==7):
        p7 = mini_strassen(a - c, e + f)
        comm.send(p7,0,rank)

    comm.barrier()

    if (rank==0):
        c11 = p5 + p4 - p2 + p6
        c12 = p1 + p2
        c21 = p3 + p4
        c22 = p1 + p5 - p3 - p7

        c = np.vstack((np.hstack((c11, c12)), np.hstack((c21, c22)))) 
        return c


if __name__ == "__main__":
    n = 0
    m1 = m2 = []
    product = []
    if (rank == 0):
        try:
            print("Size for the (n x n) matrix? ")
            n = int(input())
            if n % 2 != 0 or not (n & (n - 1) == 0):
                print("Size must be even and of a power of 2 for Strassen's algorithm. Please enter another number.")
                exit(1)
            m1 = matrix_randomizer(n)
            m2 = matrix_randomizer(n)
            print(f"n = {n}")
        except ValueError:
            print("Input not integer, please repeat")
            exit(1)

    m1 = comm.bcast(m1, root=0)
    m2 = comm.bcast(m2, root=0)

    # result = main_strassen(m1, m2)
    result = strassen(n, m1, m2, rank, 0)

    if (rank == 0):
        print(f"Output: {result}")


