// internal helper functions for MPI implementation

#define OTAWA_COMPATIBLE 1

#ifndef _MPI_INTERNAL_H
#define _MPI_INTERNAL_H

#include "fgmp.h"
#include "mpi.h"
#include <assert.h>
#include <stdbool.h>
#include <stddef.h>

typedef struct {
    int used; // ==0 if unused
    void *buf;
    int count;
    MPI_Datatype datatype;
    int source;
    int tag;
} mpi_request_t; 

#define MAX_REQUESTS 4               // maximum number of parallel Irecv

extern mpi_request_t request_slot[MAX_REQUESTS];


#define ACK_FLIT 0xbadeaffe


#define MAX_TAGS        32768           // maximum number of tags (MPI_TAG_UB+1)
                                        // DO NOT CHANGE:
                                        // hardwired in the following functions


static inline int tag_from_headflit(flit_t f)
{
    return (((f)>>(sizeof(flit_t)*8-15)) & 0x7fff);
}


static inline int len_from_headflit(flit_t f) 
{
    return ((f) & ((1L<<(sizeof(flit_t)*8-15))-1));
}


static inline flit_t headflit_from_tag_and_len(int tag, int len)
{
    return (((flit_t)(tag)<<(sizeof(flit_t)*8-15)) | (len));
}


static inline cid_t cid_from_comm(MPI_Comm comm, int rank)
{
    return (comm==MPI_COMM_WORLD) ? rank : comm->group.cids[rank];
}


// return the rank of the process in the current communicator
static inline int rank_in_comm(MPI_Comm comm, cid_t cid)
{
    if (comm==MPI_COMM_WORLD) return cid;
    int i, n=comm->group.size;
    for (i=0; i<n; n++)
        if (comm->group.cids[i]==cid)
            return i;
    return -1;
}


static inline size_t sizeof_mpi_datatype(MPI_Datatype datatype)
{
    return datatype & 0xff;
}


static inline void send_raw(cid_t dest, int len, const flit_t *buf)
{
    while (len>0) {
        fgmp_send_flit(dest, *buf++);
        len -= sizeof(flit_t);
    }
}


static inline void send_acked(cid_t dest, int len, const flit_t *buf)
{
    fgmp_send_flit(dest, *buf++);
    flit_t f = fgmp_recv_flit(dest);

#ifndef OTAWA_COMPATIBLE
    assert(f==ACK_FLIT); 
#endif
     len -= sizeof(flit_t);
    while (len>0) {
        fgmp_send_flit(dest, *buf++);
        len -= sizeof(flit_t);
    } 
}


// store only a part of a flit in memory
static inline void store_flit_fraction(void *d, unsigned len, flit_t f)
{
#ifndef OTAWA_COMPATIBLE
    assert(__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__);
#endif
    flit_t m = ((flit_t)(-2)<<(8*len-1));
    *(flit_t *)d = ((*(flit_t *)d ^ f) & m) ^ f;
}


static inline void recv_raw(cid_t source, unsigned len, flit_t *buf)
{
#ifndef OTAWA_COMPATIBLE
    assert(len > 0);
#endif
    flit_t flit;
    while (len>sizeof(flit_t)) {
        *buf++ = fgmp_recv_flit(source);
        len -= sizeof(flit_t);
    }
    store_flit_fraction(buf, len, fgmp_recv_flit(source));
}


static inline void recv_acked(cid_t source, unsigned len, flit_t *buf)
{
    flit_t f, flit;
    f = fgmp_recv_flit(source);
    fgmp_send_flit(source, ACK_FLIT);

    if (len > sizeof(flit_t)) {
        *buf++ = f;
        len -= sizeof(flit_t);
        while (len>sizeof(flit_t)) {
            *buf++ = fgmp_recv_flit(source);
            len -= sizeof(flit_t);
        }

        f = fgmp_recv_flit(source);
    }
    store_flit_fraction(buf, len, f);
}


// send a flit to all other processes
static inline void broadcast_flit(MPI_Comm comm, flit_t f)
{
    int n = comm->group.size;
    int i;

    if (comm==MPI_COMM_WORLD) {
        for (i=0; i<n; i++)
            if (i!=mpi_comm_world.rank) {
                fgmp_send_flit(i, f);
            }
    }  else {
        for (i=0; i<n; i++)
            if (i!=comm->rank) {
                fgmp_send_flit(comm->group.cids[i], f);
            }
    } 
}

static inline void wait_for_ack(MPI_Comm comm)
{
    int n = comm->group.size;
    int counter = n-1; // #processes we are waiting for
    int i = 0;
    bool ready[n];
    for (i=0; i<n; i++) ready[i] = false;
    ready[comm->rank] = true;

    while (counter>0) {
        i++;
        if (i>=n) i = 0;
        if (!ready[i]) {
            cid_t c = cid_from_comm(comm, i);
            if (fgmp_probe(c)) {
				flit_t f = fgmp_recv_flit(c);
#ifndef OTAWA_COMPATIBLE
                assert(f==ACK_FLIT);
#endif
                ready[i] = true;
                counter--;
            }
        }
    }
}


static inline void gather(flit_t *buf_per_process[], unsigned lens[], unsigned total_len)
{
    cid_t max_cid = mpi_comm_world.group.size;
    cid_t i=0;

    while (total_len>0) {
        while (lens[i]==0 || fgmp_probe(i)==false) {
            i++;
            if (i>=max_cid) i=0;
        } 

        flit_t f = fgmp_recv_flit(i);

        if (lens[i]<sizeof(flit_t)) {
            store_flit_fraction(buf_per_process[i], lens[i], f);
            total_len -= lens[i];
            lens[i] = 0;
        } else {
            *buf_per_process[i]++ = f;
            total_len -= sizeof(flit_t);
            lens[i] -= sizeof(flit_t);
        } 
    } 
}


static inline void memcopy_safe(void *dest, void*src, size_t n)
{
    int i;

    // Typecast src and dest addresses to (char *)
    char *csrc = (char *)src;
    char *cdest = (char *)dest;

    // Copy contents of src[] to dest[]
    for (i=0; i<n; i++)
        cdest[i] = csrc[i];
}

#endif

