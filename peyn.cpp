#include <mpi.h>
#include <bits/stdc++.h>

using namespace std;

// Function to print an n x n matrix
void print(int n, int** mat)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            cout << mat[i][j] << " ";
        }
        cout << endl;
    }
    cout << endl;
}

// Function to allocate memory for an n x n matrix
int** allocateMatrix(int n)
{
    int* data = (int*)malloc(n * n * sizeof(int)); // Allocate a contiguous block of memory
    int** array = (int**)malloc(n * sizeof(int*)); // Allocate an array of row pointers
    for (int i = 0; i < n; i++)
    {
        array[i] = &(data[n * i]); // Assign each row pointer to the appropriate location in the contiguous block
    }
    return array;
}

// Function to fill an n x n matrix with random values
void fillMatrix(int n, int**& mat)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            mat[i][j] = rand() % 5; // Fill matrix with random values between 0 and 4
        }
    }
}

// Function to free the memory allocated for an n x n matrix
void freeMatrix(int n, int** mat)
{
    free(mat[0]); // Free the contiguous block of memory
    free(mat);    // Free the array of row pointers
}

// Function to perform naive matrix multiplication of two n x n matrices
int** naive(int n, int** mat1, int** mat2)
{
    int** prod = allocateMatrix(n); // Allocate memory for the product matrix

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            prod[i][j] = 0;
            for (int k = 0; k < n; k++)
            {
                prod[i][j] += mat1[i][k] * mat2[k][j]; // Compute the dot product of row i and column j
            }
        }
    }

    return prod;
}

// Function to extract a submatrix (slice) of size n/2 x n/2 from mat, starting at (offseti, offsetj)
int** getSlice(int n, int** mat, int offseti, int offsetj)
{
    int m = n / 2;
    int** slice = allocateMatrix(m); // Allocate memory for the slice
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < m; j++)
        {
            slice[i][j] = mat[offseti + i][offsetj + j]; // Copy the appropriate elements
        }
    }
    return slice;
}

// Function to add or subtract two n x n matrices
int** addMatrices(int n, int** mat1, int** mat2, bool add)
{
    int** result = allocateMatrix(n); // Allocate memory for the result matrix
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            if (add)
                result[i][j] = mat1[i][j] + mat2[i][j]; // Add corresponding elements
            else
                result[i][j] = mat1[i][j] - mat2[i][j]; // Subtract corresponding elements
        }
    }

    return result;
}

// Function to combine four submatrices into one larger matrix
int** combineMatrices(int m, int** c11, int** c12, int** c21, int** c22)
{
    int n = 2 * m;
    int** result = allocateMatrix(n); // Allocate memory for the result matrix

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            if (i < m && j < m)
                result[i][j] = c11[i][j]; // Top-left submatrix
            else if (i < m)
                result[i][j] = c12[i][j - m]; // Top-right submatrix
            else if (j < m)
                result[i][j] = c21[i - m][j]; // Bottom-left submatrix
            else
                result[i][j] = c22[i - m][j - m]; // Bottom-right submatrix
        }
    }

    return result;
}

// Strassen's matrix multiplication algorithm
int** strassen(int n, int** mat1, int** mat2)
{
    // Base case: if matrix size is small, use naive multiplication
    if (n <= 32)
    {
        return naive(n, mat1, mat2);
    }

    int m = n / 2;

    // Divide matrices into 4 submatrices each
    int** a = getSlice(n, mat1, 0, 0);
    int** b = getSlice(n, mat1, 0, m);
    int** c = getSlice(n, mat1, m, 0);
    int** d = getSlice(n, mat1, m, m);
    int** e = getSlice(n, mat2, 0, 0);
    int** f = getSlice(n, mat2, 0, m);
    int** g = getSlice(n, mat2, m, 0);
    int** h = getSlice(n, mat2, m, m);

    // Calculate the 7 products using Strassen's formulas
    int** p1 = strassen(m, addMatrices(m, a, d, true), addMatrices(m, e, h, true));
    int** p2 = strassen(m, addMatrices(m, c, d, true), e);
    int** p3 = strassen(m, a, addMatrices(m, f, h, false));
    int** p4 = strassen(m, d, addMatrices(m, g, e, false));
    int** p5 = strassen(m, addMatrices(m, a, b, true), h);
    int** p6 = strassen(m, addMatrices(m, c, a, false), addMatrices(m, e, f, true));
    int** p7 = strassen(m, addMatrices(m, b, d, false), addMatrices(m, g, h, true));

    // Combine the 7 products to get the final quadrants of the product matrix
    int** c11 = addMatrices(m, addMatrices(m, addMatrices(m, p1, p4, true), p7, false), p5, true);
    int** c12 = addMatrices(m, p3, p5, true);
    int** c21 = addMatrices(m, p2, p4, true);
    int** c22 = addMatrices(m, addMatrices(m, addMatrices(m, p1, p3, false), p2, true), p6, true);

    // Combine the quadrants into the final product matrix
    int** prod = combineMatrices(m, c11, c12, c21, c22);

    // Free memory used for submatrices and temporary products
    freeMatrix(m, a);
    freeMatrix(m, b);
    freeMatrix(m, c);
    freeMatrix(m, d);
    freeMatrix(m, e);
    freeMatrix(m, f);
    freeMatrix(m, g);
    freeMatrix(m, h);

    freeMatrix(m, p1);
    freeMatrix(m, p2);
    freeMatrix(m, p3);
    freeMatrix(m, p4);
    freeMatrix(m, p5);
    freeMatrix(m, p6);
    freeMatrix(m, p7);

    freeMatrix(m, c11);
    freeMatrix(m, c12);
    freeMatrix(m, c21);
    freeMatrix(m, c22);

    return prod;
}

// MPI-based parallel version of Strassen's algorithm
void strassenMPI(int n, int** mat1, int** mat2, int**& prod, int rank, int numProcs)
{
    if (n <= 32)
    {
        prod = naive(n, mat1, mat2);
        return;
    }

    int m = n / 2;

    int** a = getSlice(n, mat1, 0, 0);
    int** b = getSlice(n, mat1, 0, m);
    int** c = getSlice(n, mat1, m, 0);
    int** d = getSlice(n, mat1, m, m);
    int** e = getSlice(n, mat2, 0, 0);
    int** f = getSlice(n, mat2, 0, m);
    int** g = getSlice(n, mat2, m, 0);
    int** h = getSlice(n, mat2, m, m);

    int** p1;
    int** p2;
    int** p3;
    int** p4;
    int** p5;
    int** p6;
    int** p7;

    if (rank == 0)
    {
        MPI_Send(&a[0][0], m * m, MPI_INT, 1, 0, MPI_COMM_WORLD);
        MPI_Send(&e[0][0], m * m, MPI_INT, 1, 0, MPI_COMM_WORLD);

        MPI_Send(&c[0][0], m * m, MPI_INT, 2, 0, MPI_COMM_WORLD);
        MPI_Send(&e[0][0], m * m, MPI_INT, 2, 0, MPI_COMM_WORLD);

        MPI_Send(&a[0][0], m * m, MPI_INT, 3, 0, MPI_COMM_WORLD);
        MPI_Send(&f[0][0], m * m, MPI_INT, 3, 0, MPI_COMM_WORLD);

        p4 = strassen(m, d, addMatrices(m, g, e, false));

        MPI_Recv(&p1[0][0], m * m, MPI_INT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&p2[0][0], m * m, MPI_INT, 2, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&p3[0][0], m * m, MPI_INT, 3, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    else if (rank == 1)
    {
        int** a_temp = allocateMatrix(m);
        int** e_temp = allocateMatrix(m);

        MPI_Recv(&a_temp[0][0], m * m, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&e_temp[0][0], m * m, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        p1 = strassen(m, addMatrices(m, a_temp, d, true), addMatrices(m, e_temp, h, true));

        MPI_Send(&p1[0][0], m * m, MPI_INT, 0, 0, MPI_COMM_WORLD);

        freeMatrix(m, a_temp);
        freeMatrix(m, e_temp);
    }
    else if (rank == 2)
    {
        int** c_temp = allocateMatrix(m);
        int** e_temp = allocateMatrix(m);

        MPI_Recv(&c_temp[0][0], m * m, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&e_temp[0][0], m * m, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        p2 = strassen(m, addMatrices(m, c_temp, d, true), e_temp);

        MPI_Send(&p2[0][0], m * m, MPI_INT, 0, 0, MPI_COMM_WORLD);

        freeMatrix(m, c_temp);
        freeMatrix(m, e_temp);
    }
    else if (rank == 3)
    {
        int** a_temp = allocateMatrix(m);
        int** f_temp = allocateMatrix(m);

        MPI_Recv(&a_temp[0][0], m * m, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&f_temp[0][0], m * m, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        p3 = strassen(m, a_temp, addMatrices(m, f_temp, h, false));

        MPI_Send(&p3[0][0], m * m, MPI_INT, 0, 0, MPI_COMM_WORLD);

        freeMatrix(m, a_temp);
        freeMatrix(m, f_temp);
    }

    // Combine the 7 products to get the final quadrants of the product matrix
    if (rank == 0)
    {
        int** c11 = addMatrices(m, addMatrices(m, addMatrices(m, p1, p4, true), p7, false), p5, true);
        int** c12 = addMatrices(m, p3, p5, true);
        int** c21 = addMatrices(m, p2, p4, true);
        int** c22 = addMatrices(m, addMatrices(m, addMatrices(m, p1, p3, false), p2, true), p6, true);

        prod = combineMatrices(m, c11, c12, c21, c22);

        freeMatrix(m, p1);
        freeMatrix(m, p2);
        freeMatrix(m, p3);
        freeMatrix(m, p4);
        freeMatrix(m, p5);
        freeMatrix(m, p6);
        freeMatrix(m, p7);

        freeMatrix(m, c11);
        freeMatrix(m, c12);
        freeMatrix(m, c21);
        freeMatrix(m, c22);
    }

    freeMatrix(m, a);
    freeMatrix(m, b);
    freeMatrix(m, c);
    freeMatrix(m, d);
    freeMatrix(m, e);
    freeMatrix(m, f);
    freeMatrix(m, g);
    freeMatrix(m, h);
}

int main(int argc, char* argv[])
{
    int rank, numProcs;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

    int n = 128; // Size of the matrices (must be a power of 2)
    int** mat1 = allocateMatrix(n);
    int** mat2 = allocateMatrix(n);
    int** prod;

    if (rank == 0)
    {
        fillMatrix(n, mat1);
        fillMatrix(n, mat2);
    }

    strassenMPI(n, mat1, mat2, prod, rank, numProcs);

    if (rank == 0)
    {
        print(n, prod);
        freeMatrix(n, prod);
    }

    freeMatrix(n, mat1);
    freeMatrix(n, mat2);

    MPI_Finalize();
    return 0;
}
