import numpy as np
import random

def matrix_randomizer(n):
    matrix = np.random.randint(0, 11, size=(n, n))
    print(f"matrix: {matrix}")
    return matrix

def split(matrix):
    row, col = matrix.shape
    row2, col2 = row//2, col//2
    return matrix[:row2, :col2], matrix[:row2, col2:], matrix[row2:, :col2], matrix[row2:, col2:]

def strassen(x, y):
    if len(x) == 1:
        return x * y

    a, b, c, d = split(x)
    e, f, g, h = split(y)

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

    c = np.vstack((np.hstack((c11, c12)), np.hstack((c21, c22)))) 

    return c


if __name__ == "__main__":
    n = 50
    print("Size for the (n x n) matrix? ")
    try:
        n = int(input())
        if n % 2 != 0:
            print("Size must be even for Strassen's algorithm. Please enter an even number.")
            exit(1)
        m1 = matrix_randomizer(n)
        m2 = matrix_randomizer(n)
    except ValueError:
        print("Input not integer, please repeat")
        exit(1)
    print(f"n = {n}")

    result = strassen(m1, m2)

    print(f"Output: {result}")


