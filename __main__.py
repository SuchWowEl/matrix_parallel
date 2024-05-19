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

    # Phase 1: Determine which submatrices each process needs to receive and perform multiplication
    send_recv_pairs = {
        0: [4, 5],  # Process 0 receives from 4 and sends to 5
        1: [6, 7],  # Process 1 receives from 6 and sends to 7
        2: [5, 4],  # Process 2 receives from 5 and sends to 4
        3: [7, 6],  # Process 3 receives from 7 and sends to 6
        4: [2, 0],  # Process 4 receives from 2 and sends to 0
        5: [0, 2],  # Process 5 receives from 0 and sends to 2
        6: [3, 1],  # Process 6 receives from 3 and sends to 1
        7: [1, 3]   # Process 7 receives from 1 and sends to 3
    }

    # If the current rank has a pair in the dictionary
    if rank in send_recv_pairs:
        # Get the ranks to receive from and send to
        receive_from, send_to = send_recv_pairs[rank]

        if rank % 2 == 0:  # If the rank is even
            # Send localmat to the rank we're supposed to send to
            comm.send(localmat, dest=send_to, tag=rank)
            # Receive the matrix from the rank we're supposed to receive from
            temp = comm.recv(source=receive_from, tag=receive_from)
        else:  # If the rank is odd
            # Receive the matrix from the rank we're supposed to receive from
            temp = comm.recv(source=receive_from, tag=receive_from)
            # Send localmat to the rank we're supposed to send to
            comm.send(localmat, dest=send_to, tag=rank)

        # Perform the multiplication and add the result to temp
        temp = multi(temp, localmat)
    comm.Barrier()



    # Phase 2: Combine partial results from the previous step
    phase_2_pairs = {
        0: 1,  # Process 0 needs to add results from process 1
        5: 7,  # Process 5 needs to add results from process 7 
        4: 6,  # Process 4 needs to add results from process 6
        2: 3  # Process 2 needs to add results from process 3
    }

    # If the current process rank is a key in phase_2_pairs
    if rank in phase_2_pairs:
        # Get the rank to receive from
        other_rank = phase_2_pairs[rank]
        
        # Receive the matrix from the rank we're supposed to receive from
        localmat = comm.recv(source=other_rank, tag=other_rank)
        
        # Add the received matrix to temp
        temp += localmat

    # If the current process rank is a value in phase_2_pairs
    elif rank in phase_2_pairs.values():
        # Get the rank to send to
        send_to = list(phase_2_pairs.keys())[list(phase_2_pairs.values()).index(rank)]
        
        # Send temp to the rank we're supposed to send to
        comm.send(temp, dest=send_to, tag=rank)

    return temp, list(phase_2_pairs.keys())

# Example usage:
if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if size != 8:
        raise ValueError("This program requires exactly 8 MPI processes.")

    matrix_size = 8
    localmat = populate_matrices(matrix_size, rank)

    comm.Barrier()  # Synchronize all processes

    if rank == 0:
        tb = time.time()

    localmat, c_keys = startitall(localmat, matrix_size, rank, comm)
    comm.Barrier()

    if rank == 0:
        ta = time.time()
        print(f"time for submatrix generation: {ta-tb}")

    if rank in c_keys:
        print(f"Process {rank} result: {localmat}")

    comm.Barrier()

    if rank == 0:
        print("\nFinal Result Matrix C:")
    for key in c_keys:
        if rank == key:
            print(f"Process {rank}: {localmat}")
