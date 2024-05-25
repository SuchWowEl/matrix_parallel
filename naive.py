import numpy as np
import time

# NOTE:
# pmultiply runtime:
# 256 = 0.044034481048583984
# 512 = 0.434398889541626
# 1024 = 4.980994701385498
# 2048 = 70.91712617874146
size = 2048

def multiply(A, B, C):
    n = size
    for i in range(n):
        for j in range( n):
            C[i][j] = 0
            for k in range(n):
                C[i][j] += A[i][k]*B[k][j]

def pmultiply(A, B, C):
    C = np.matmul(A,B)
# this code is contributed by shivanisinghss2110

a = np.random.randint(0, 11, size=(size, size))
b = np.random.randint(0, 11, size=(size, size))
c = np.zeros((size, size), dtype=int)

ta = time.time()
pmultiply(a,b,c)
tb = time.time()

print(f"time elapsed: {tb-ta}")
