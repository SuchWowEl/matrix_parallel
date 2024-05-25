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
        # A
        0: [4, 5],  # Process 0 receives from 4 and sends to 5. 0(4) or A11(B11)
        1: [6, 7],  # Process 1 receives from 6 and sends to 7. 1(6) or A12(B21)
        2: [5, 4],  # Process 2 receives from 5 and sends to 4. 2(5) or A21(B12)
        3: [7, 6],  # Process 3 receives from 7 and sends to 6. 3(7) or A22(B22)

        # B. Since we're computing for AB and not BA, values are switched during multiplication
        # for processes 4 to 7. So that instead of BA(localmatxtemp), it's AB(tempxlocalmat). Refer to line 57
        4: [2, 0],  # Process 4 receives from 2 and sends to 0. 4(2) or B11(A21)
        5: [0, 2],  # Process 5 receives from 0 and sends to 2. 5(0) or B12(A11)
        6: [3, 1],  # Process 6 receives from 3 and sends to 1. 6(3) or B21(A22)
        7: [1, 3]   # Process 7 receives from 1 and sends to 3. 7(1) or B22(A12)
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

        print(f"This is process {rank}, my temp before multi is: {temp}")

        if rank < 4:
            # Perform the multiplication and add the result to temp, reusing it to save space
            temp = multi(localmat, temp)
        else:
            # Perform the multiplication and add the result to temp, switch values if rank is greater than 3. Since BA is not the same as AB
            temp = multi(temp, localmat)
    comm.Barrier()

    print(f"\n\nThis is process {rank}, and my multi result is: {temp}")

    # Phase 2: Combine partial results from the previous step
    phase_2_pairs = {
        # C
        0: 1,  # Process 0 needs to add results from process 1. This is C11
        5: 7,  # Process 5 needs to add results from process 7. This is C12
        4: 6,  # Process 4 needs to add results from process 6. This is C21
        2: 3   # Process 2 needs to add results from process 3. This is C22
    }

    # If the current process rank is a key in phase_2_pairs
    if rank in phase_2_pairs:
        # Get the rank to receive from
        other_rank = phase_2_pairs[rank]
        
        # Receive the matrix from the rank we're supposed to receive from into localmat, since the values inside it are no longer needed for the addition
        localmat = comm.recv(source=other_rank, tag=other_rank)
        
        # Add the received matrix to temp
        temp += localmat

    # If the current process rank is a value in phase_2_pairs
    elif rank in phase_2_pairs.values():
        # Get the rank to send to
        send_to = list(phase_2_pairs.keys())[list(phase_2_pairs.values()).index(rank)]
        
        # Send temp to the rank we're supposed to send to
        comm.send(temp, dest=send_to, tag=rank)

    return temp

# Example usage:
if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if size != 8:
        raise ValueError("This program requires exactly 8 MPI processes.")

    # NOTE:
    # runtime (el):
    # 512 = 0.07595205307006836
    # 1024 = 0.5858025550842285
    # 2048 = 15.199561357498169
    # 4096 = 225.06619834899902
    # 8192 = 1877.4816830158234
    matrix_size = 4
    localmat = populate_matrices(matrix_size, rank) # Each process creates their own local matrix named localmat
    """
    For convention, the matrix A is made up of the matrices from processes 0 to 3. Essentially, process 0 has A11, 1 has A12
    2 has A21, and process 3 has A22. Similarly, the matrix B is divided among processes 4 to 7.
    """

    comm.Barrier()  # Synchronize all processes

    if rank == 0:
        ta = time.time()

    localmat = startitall(localmat, matrix_size, rank, comm)
    comm.Barrier()

    if rank == 0:
        tb = time.time()
        print(f"time elapsed: {ta-tb}")
        print("\nFinal Result Matrix C:")

    # c_submatrices lists the processes and their corresponding placement in the matrix C, or the product matrix.
    c_submatrices = {
        0: "C11",
        2: "C22",
        4: "C21",
        5: "C12",
    }
    if rank in c_submatrices.keys():
        print(f"{c_submatrices[rank]}: {localmat}")
