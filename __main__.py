from mpi4py import MPI
import random

comm = MPI.COMM_WORLD
rank = comm.rank

def matrix_randomizer(n):
    matrix = []
    for i in range(n):
        x = []
        for j in range(n):
            x.append(random.randint(0, 10))
        matrix.append(x)
    print(f"matrix: {matrix}")
    return matrix

def strassen(n, m1, m2, prod, rank):
    if (n==1):
        prod[0][0] = m1[0][0] * m2[0][0]

    # TODO: check if // usage is correct
    h = n // 2

if __name__ == "__main__":
    n = 0
    m1 = m2 = []
    product = []
    if (rank == 0):
        print("Size for the (n x n) matrix? ")
        try:
            n = int(input())
            m1 = matrix_randomizer(n)
            m2 = matrix_randomizer(n)
        except ValueError:
            print("Input not integer, please repeat")
            exit(1)
        print(f"n = {n}")

    m1 = comm.bcast(m1, root=0)
    m2 = comm.bcast(m2, root=0)

    comm.Barrier()
    if (rank!=0):
        print(f"process: {rank} \n{m1} \n{m2}")
    strassen(n, m1, m2, product, rank)
