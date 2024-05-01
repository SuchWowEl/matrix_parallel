from mpi4py import MPI
import numpy as np
import time

comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size

def matrix_randomizer(n):
    matrix = np.random.randint(0, 11, size=(n, n))
    print(f"matrix: {matrix}")
    return matrix

def split(matrix):
    row, col = matrix.shape
    row2, col2 = row//2, col//2
    return matrix[:row2, :col2], matrix[:row2, col2:], matrix[row2:, :col2], matrix[row2:, col2:]

def naive(mat1, mat2):
    # Allocate matrix for product
    return mat1 @ mat2

def strassen_scatter(m1, m2):
    if len(m1) == 1:
        return m1 * m2

    a, b, c, d = split(m1)
    e, f, g, h = split(m2)

    return ((a, f - h), (a + b, h), (c + d, e), (d, g - e), (a + d, e + h), (b - d, g + h), (a - c, e + f))

def strassen_gather(p1, p2, p3, p4, p5, p6, p7):
    c11 = p5 + p4 - p2 + p6
    c12 = p1 + p2
    c21 = p3 + p4
    c22 = p1 + p5 - p3 - p7

    return np.vstack((np.hstack((c11, c12)), np.hstack((c21, c22))))

def strassen(m1, m2):
    if (n <= 1):
        return naive(m1, m2)
    # if len(m1) == 1:
    #     return m1 * m2

    a, b, c, d = split(m1)
    e, f, g, h = split(m2)

    p1 = strassen(a, f - h)
    p2 = strassen(a + b, h)
    p3 = strassen(c + d, e)
    p4 = strassen(d, g - e)
    p5 = strassen(a + d, e + h)
    p6 = strassen(b - d, g + h)
    p7 = strassen(a - c, e + f)

    c11 = p5 + p4 - p2 + p6
    c12 = p1 + p2
    c21 = p3 + p4
    c22 = p1 + p5 - p3 - p7

    return np.vstack((np.hstack((c11, c12)), np.hstack((c21, c22))))

if __name__ == "__main__":
    n = 0
    m1 = m2 = []

    if rank == 0:
        try:
            print("Size for the (n x n) matrix? ")
            n = int(input())
            if n % 2 != 0 or not (n & (n - 1) == 0):
                print("Size must be even and of a power of 2 for Strassen's algorithm. Please enter another number.")
                exit(1)
            else:
                print(f"n = {n}")
            m1 = matrix_randomizer(n)
            m2 = matrix_randomizer(n)
        except ValueError:
            print("Input not integer, please repeat")
            exit(1)


    t1 = time.time()
    if rank == 0:
        workloads = strassen_scatter(m1, m2)
        for i in range(len(workloads)):
            comm.send((workloads[i][0],workloads[i][1]), i+1, i+1)
        print("1st send")

    else:
        sub_matrix = comm.recv(source=0,tag=rank)
        sub_result = strassen(*sub_matrix)
        comm.send(sub_result, 0, rank)
        print("over barrier")

    # comm.barrier()

    p1 = comm.recv(source=1,tag=1)
    p2 = comm.recv(source=2,tag=2)
    p3 = comm.recv(source=3,tag=3)
    p4 = comm.recv(source=4,tag=4)
    p5 = comm.recv(source=5,tag=5)
    p6 = comm.recv(source=6,tag=6)
    p7 = comm.recv(source=7,tag=7)
    gathered_results = (p1,p2,p3,p4,p5,p6,p7)
    print(strassen_gather(*gathered_results))
    t2 = time.time()

    print("Time: ", t2-t1)

