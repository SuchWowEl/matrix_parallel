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

# TODO: Confirm if this function is pass-by-reference.
#  It becomes like that when you apparently pass a list to it.
def matrix_slicer(o):
    # m = n // 2
    sub_mat = [[o["matrix"][o["offset_row"] + i][o["offset_col"] + j] for j in range(o["hn"])] for i in range(o["hn"])]
    print(f"{o['offset_row']}{o['offset_col']}: {sub_mat}")
    return sub_mat

# TODO: confirm if this function is pass-by-reference
def strassen(o):
    if (n==1):
        o["prod"][0][0] = m1[0][0] * m2[0][0]

    # NOTE: // usage might not be correct
    hn = n // 2

    a = matrix_slicer({"n": 4, "matrix": o["m1"], "offset_row": 0, "offset_col": 0, "hn": hn})
    b = matrix_slicer({"n": 4, "matrix": o["m1"], "offset_row": 0, "offset_col": hn, "hn": hn})
    c = matrix_slicer({"n": 4, "matrix": o["m1"], "offset_row": hn, "offset_col": 0, "hn": hn})
    d = matrix_slicer({"n": 4, "matrix": o["m1"], "offset_row": hn, "offset_col": hn, "hn": hn})
    e = matrix_slicer({"n": 4, "matrix": o["m2"], "offset_row": 0, "offset_col": 0, "hn": hn})
    f = matrix_slicer({"n": 4, "matrix": o["m2"], "offset_row": 0, "offset_col": hn, "hn": hn})
    g = matrix_slicer({"n": 4, "matrix": o["m2"], "offset_row": hn, "offset_col": 0, "hn": hn})
    h = matrix_slicer({"n": 4, "matrix": o["m2"], "offset_row": hn, "offset_col": hn, "hn": hn})

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

    if (rank!=0):
        print(f"process: {rank} \n{m1} \n{m2}")
    strassen({"n": n, "m1": m1, "m2": m2, "prod": product, "rank": rank})
