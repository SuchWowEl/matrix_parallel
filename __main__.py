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
        0: [4, 5],  # Process 0 needs B11 (4) and B12 (5)
        1: [6, 7],  # Process 1 needs B21 (6) and B22 (7)
        2: [4, 5],  # Process 2 needs B11 (4) and B12 (5)
        3: [6, 7],  # Process 3 needs B21 (6) and B22 (7)
        4: [0, 2],  # Process 4 needs A11 (0) and A21 (2)
        5: [0, 2],  # Process 5 needs A11 (0) and A21 (2)
        6: [1, 3],  # Process 6 needs A12 (1) and A22 (3)
        7: [1, 3]   # Process 7 needs A12 (1) and A22 (3)
    }

    for i in range(len(send_recv_pairs[rank])):
        receive_from = send_recv_pairs[rank][i]
        comm.send(localmat, dest=send_recv_pairs[rank][(i + 1) % len(send_recv_pairs[rank])], tag=rank)
        received_mat = comm.recv(source=receive_from, tag=receive_from)
        temp += multi(localmat, received_mat)
    comm.Barrier()

    # Phase 2: Combine partial results from the previous step
    phase_2_pairs = {
        0: 1,  # Process 0 needs to add results from process 1
        2: 3,  # Process 2 needs to add results from process 3
        4: 5,  # Process 4 needs to add results from process 5
        6: 7   # Process 6 needs to add results from process 7
    }

    if rank in phase_2_pairs:
        other_rank = phase_2_pairs[rank]
        received_mat = comm.recv(source=other_rank, tag=other_rank)
        temp += received_mat
    elif rank in phase_2_pairs.values():
        comm.send(temp, dest=list(phase_2_pairs.keys())[list(phase_2_pairs.values()).index(rank)], tag=rank)

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
