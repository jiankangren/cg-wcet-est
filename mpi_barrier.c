#include "mpi_internal.h"

int MPI_Barrier(MPI_Comm comm)
{
    int next_process;
#ifndef OTAWA_COMPATIBLE
    assert(comm != MPI_COMM_NULL);
#endif

    int my_rank;// = comm->rank;
    if (my_rank==0) {
        // send ack to next process
        fgmp_send_flit(cid_from_comm(comm, 1), 0);
        // wait for ack from last process
        flit_t f = fgmp_recv_flit(cid_from_comm(comm, comm->group.size-1));
#ifndef OTAWA_COMPATIBLE
        assert(f==(comm->group.size-1));
#endif
        // tell all
        broadcast_flit(comm, 42);
    } else {
        // wait for ack from previous process
       flit_t f = fgmp_recv_flit(cid_from_comm(comm, my_rank-1));
#ifndef OTAWA_COMPATIBLE
        assert(f==(my_rank-1));
#endif
        // send ack to next process
        // 1. compute next process; former code: (my_rank+1)%comm->group.size
        //    because % is unbounded avoid it!
        next_process = my_rank + 1;
        if (next_process > comm->group.size) next_process = 0;
        // 2. send flit
        fgmp_send_flit(cid_from_comm(comm, next_process), my_rank);  
        // wait for broadcast from process 0
        f = fgmp_recv_flit(cid_from_comm(comm, 0)); 
#ifndef OTAWA_COMPATIBLE
        assert(f==42); 
#endif 
    } 
    return MPI_SUCCESS;
}

// Alternative, centralised implementation
int MPI_Barrier_Alternative(MPI_Comm comm)
{
#ifndef OTAWA_COMPATIBLE
    assert(comm != MPI_COMM_NULL);
#endif
    if (comm->rank!=0) {
        cid_t root = cid_from_comm(comm, 0);
        flit_t f = fgmp_recv_flit(root);
#ifndef OTAWA_COMPATIBLE
        assert(f==1);
#endif
        fgmp_send_flit(root, ACK_FLIT);
        f = fgmp_recv_flit(root);
#ifndef OTAWA_COMPATIBLE
        assert(f==2);
#endif
    } else {
        // tell other processes that root is ready
        broadcast_flit(comm, 1);
        wait_for_ack(comm);
        broadcast_flit(comm, 2);
    }
    return MPI_SUCCESS;
}
