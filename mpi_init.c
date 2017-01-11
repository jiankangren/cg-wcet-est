#include "mpi_internal.h"


mpi_request_t request_slot[MAX_REQUESTS];
mpi_communicator_t mpi_comm_world;

int MPI_Init(int *argc, char ***argv)
{
    // for nonblocking receive
    int i;
    for (i=0; i<MAX_REQUESTS; i++)
        request_slot[i].used = 0;

    mpi_comm_world.rank = fgmp_get_cid();
    
    mpi_comm_world.group.size = fgmp_get_max_cid();

    for (i=0; i<255; i++) {
        mpi_comm_world.group.cids[i] = 0;
    }

    return MPI_SUCCESS;
}
