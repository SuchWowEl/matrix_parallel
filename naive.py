import numpy as np
import time

size = 128

def multiply(A, B, C):
    n = size
    for i in range(n):
        for j in range( n):
            C[i][j] = 0
            for k in range(n):
                C[i][j] += A[i][k]*B[k][j]

# this code is contributed by shivanisinghss2110

a = np.random.randint(0, 11, size=(size, size))
b = np.random.randint(0, 11, size=(size, size))
c = np.zeros((size, size), dtype=int)

ta = time.time()
multiply(a,b,c)
tb = time.time()

print(f"time elapsed: {tb-ta}")
