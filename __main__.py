from mpi4py import MPI
import numpy as np
import time

def populate_matrices(matrix_size, rank):
    size = matrix_size // 2
    # Populate matrix A or B depending on the rank
    mat = np.random.randint(0, 11, size=(size, size))
    print(f"This is process {rank}, my mat is: {mat}")
    return mat

def multi(a, b):
    # Perform matrix multiplication using np.matmul and return directly
    return np.matmul(a, b)

def startitall(localmat, mat_size, rank, comm):
    loc_size = mat_size // 2
    temp = np.zeros((loc_size, loc_size), dtype=int)
    # Mapping of processes to submatrices
    # A11, A12, A21, A22, B11, B12, B21, B22
    submat_mapping = {
        0: "A11", 1: "A12", 2: "A21", 3: "A22",
        4: "B11", 5: "B12", 6: "B21", 7: "B22"
    }

    # Determine which submatrices each process needs to receive
    send_recv_pairs = {
        0: [4, 5],  # Process 0 needs B11 (4) and sends to (5) 
        1: [6, 7],  # Process 1 needs B21 (6) and sends to (7) 
        2: [5, 4],  # Process 2 needs B12 (5) and sends to (4) 
        3: [7, 6],  # Process 3 needs B22 (7) and sends to (6) 
        4: [2, 0],  # Process 4 needs A21 (2) and sends to (0) 
        5: [0, 2],  # Process 5 needs A11 (0) and sends to (2) 
        6: [3, 1],  # Process 6 needs A22 (3) and sends to (1) 
        7: [1, 3]   # Process 7 needs A12 (1) and sends to (3) 
    }

    submatrix_needed = send_recv_pairs[rank]
    receive_from = submatrix_needed[i]  # Get the rank to receive from
    send_to = submatrix_needed[(i + 1) % len(submatrix_needed)]  # Get the rank to send to

    #Phase 1
    for i in range(len(submatrix_needed)):
        comm.send(localmat, dest=send_to, tag=rank) #Send processes' local matrix to where it needs to go
        localmat = comm.recv(source=receive_from, tag=receive_from) #Get matrix needed for multiplication
    comm.Barrier()
    
    temp = multi(localmat, temp) #Multiply matrices and save result in temp
    comm.Barrier()
    
    #Phase 2
    # Determine which submatrices each process needs to receive
    send_recv_pairs = {
        0: [1],  # Process 0 needs result from (1) 
        5: [7],  # Process 5 needs result from (7)  
        4: [6],  # Process 4 needs result from (6)  
        2: [3],  # Process 2 needs result from (3) 
    }

    if rank < 4:
        if rank % 2 == 0:   #Processes 0 and 2 receive from 1 and 3 respectively
            localmat = comm.recv(source=(rank+1), tag=rank)
        else:
            comm.send(temp, dest=(rank-1), tag=(rank-1))
    elif rank >= 4:
        if rank == 4 or 5: #Processes 4 and 5 receive from 6 and 7 respectively
            localmat = comm.recv(source=(rank+2), tag=rank)
        else:
            comm.send(temp, dest=(rank-2), tag=(rank-2))

    if rank in send_recv_pairs:
        temp = temp + localmat  #This is where addition comes into play, this obviously don't work so....fix it

# Example usage:
if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    matrix_size = 8
    localmat = []

    tb = 0
    if rank == 0:
        tb = time.time()
    localmat = populate_matrices(matrix_size, rank)

    comm.Barrier()  # Synchronize all processes

    if rank == 0:
        ta = time.time()
        print(f"time for submatrix generation: {ta-tb}")

    # Print the matrices
    for i in range(4):
        if rank == i and rank % 2 == 0:
            print(f"\nM1: {localmat}")
        elif rank == i:
            print(f" {localmat}")
    
    for i in range(4, 8):
        if rank == i and rank % 2 == 0:
            print(f"\nM2: {localmat}")
        elif rank == i:
            print(f" {localmat}")

    comm.Barrier()

    startitall(localmat, matrix_size, rank, comm)
