/*************************************************************************/
/*                                                                       */
/*        N  A  S     P A R A L L E L     B E N C H M A R K S  3.3       */
/*                                                                       */
/*                                   C G                                 */
/*                                                                       */
/*************************************************************************/
/*    ####                   original message:                  ####     */
/*    This benchmark is part of the NAS Parallel Benchmark 3.3 suite.    */
/*    It is described in NAS Technical Reports 95-020 and 02-007         */
/*                                                                       */
/*    Permission to use, copy, distribute and modify this software       */
/*    for any purpose with or without fee is hereby granted.  We         */
/*    request, however, that all derived work reference the NAS          */
/*    Parallel Benchmarks 3.3. This software is provided "as is"         */
/*    without express or implied warranty.                               */
/*                                                                       */
/*    Information on NPB 3.3, including the technical report, the        */
/*    original specifications, source code, results and information      */
/*    on how to submit new results, is available at:                     */
/*                                                                       */
/*    ####                   change message:                    ####     */
/*    The original version of this benchmark was written in Fortran.     */
/*    This is the C port of this benchmark.                              */
/*                                                                       */
/*           http://www.nas.nasa.gov/Software/NPB/                       */
/*                                                                       */
/*    Send comments or suggestions to  npb@nas.nasa.gov                  */
/*                                                                       */
/*          NAS Parallel Benchmarks Group                                */
/*          NASA Ames Research Center                                    */
/*          Mail Stop: T27A-1                                            */
/*          Moffett Field, CA   94035-1000                               */
/*                                                                       */
/*          E-mail:  npb@nas.nasa.gov                                    */
/*          Fax:     (650) 604-3957                                      */
/*                                                                       */
/*************************************************************************/


/*************************************************************************/
//
// Authors (Fortran):	M. Yarrow
//          			C. Kuszmaul
//          			R. F. Van der Wijngaart
//          			H. Jin
//
// Authors (C port):	R. Schmid
/*************************************************************************/


/*
 * Difference to original:
 * - uses MPI_Allreduce instead of hand-written logarithmic message exchange
 * - no longer do iteration 1 twice to warm caches and page tables
 */


#include "mpi.h"
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define DEBUG(ch) asm volatile ("csrw 0x782, %0" :: "r"(ch))




#if !defined(__riscv) && !defined(ARMV6M)
// create subset of a communicator:
// n consecutive ranks belong to one communicator
void parop_comm_split_consecutive(MPI_Comm all, unsigned n, MPI_Comm *subset)
{
    int my_rank;
    MPI_Comm_rank(all, &my_rank);
    MPI_Comm_split(all, my_rank/n, my_rank, subset);
}
#endif







double *harness_malloc_double(size_t n)
{
    double *p = malloc(n * sizeof(double));
    if (p==0) {
        printf("Out of memory\n");
        MPI_Finalize();
        exit(1);
    }
    return p;
}


int *harness_malloc_int(size_t n)
{
    int *p = malloc(n * sizeof(int));
    if (p==0) {
        printf("Out of memory\n");
        MPI_Finalize();
        exit(1);
    }
    return p;
}


bool *harness_malloc_bool(size_t n)
{
    bool *p = malloc(n * sizeof(bool));
    if (p==0) {
        printf("Out of memory\n");
        MPI_Finalize();
        exit(1);
    }
    return p;
}





uint64_t randNASseed = 314159265;

uint64_t randNAS(uint64_t *seed)
{
    return *seed = ((*seed * 1220703125) & 0x00003fffffffffff);
}

double randNAS_double()
{
    return 0x1p-46 * randNAS(&randNASseed);
} 








#define CG_ITERATIONS   15


MPI_Comm comm_nprocs; // communicator with all processes that are really used
MPI_Comm comm_nrow;
//MPI_Comm comm_ncol;



/*****************************************************************/
/*************            S P R N V C             ****************/
/*****************************************************************/
void sprnvc(int n, int nz, double *v, int *iv, int *nzloc, bool *mark)
{
/*********************************************************************/
/* generate a sparse n-vector (v, iv)								 */
/* having nzv nonzeros												 */
/* 																	 */
/* mark(i) is set to 1 if position i is nonzero.					 */
/* mark is all zero on entry and is reset to all zero before exit	 */
/* this corrects a performance bug found by John G. Lewis, caused by */
/* reinitialization of mark on every one of the n calls to sprnvc	 */
/*********************************************************************/
    int i, ii;
    int nzv = 0;
    int nzrow = 0;

    //  nn1 is the smallest power of two not less than n
    int nn1 = 1;
    do {
        nn1 = 2 * nn1;
    } while( nn1 < n );

    while (1) {
        if( nzv >= nz ) break;
        double vecelt = randNAS_double();

        // generate evenly distributed random number between 1 and n
        i = randNAS_double() * nn1 + 1;
        if (i>n) continue;

        // entry already occupied?
        if (mark[i]==0) {
            mark[i] = 1;
            nzloc[++nzrow] = i;
            v[++nzv] = vecelt;
            iv[nzv] = i;
        }
    }

    // clean mark for next use
    for (ii=1; ii<=nzrow; ii++) {
        i = nzloc[ii];
        mark[i] = 0;
    }
}


/*****************************************************************/
/*************            V E C S E T             ****************/
/*****************************************************************/
void vecset( int n, double *v, int *iv, int *nzv, int i, double val )
{
	int k;
/****************************************************************/
/*  set ith element of sparse vector (v, iv) with				*/
/*  nzv nonzeros to val											*/
/****************************************************************/	
	bool set = false;
	for( k=1; k<=(*nzv); ++k )
	{
		if( iv[k] == i )
		{
			v[k] = val;
			set = true;
		}
	}
	if( !set )
	{
		*nzv = (*nzv) + 1;
		v[*nzv] = val;
		iv[*nzv] = i;
	}
}


/*****************************************************************/
/*************            S P A R S E             ****************/
/*****************************************************************/
void sparse( 	double a[], int colidx[], int rowstr[],
				int n, int arow[], int acol[], double aelt[],
				int firstrow, int lastrow, double x[],
				int nzloc[], bool mark[], int nnza )
{
/****************************************************************************/
/* rows range from firstrow to lastrow										*/
/* the rowstr pointers are defined for nrows = lastrow-firstrow+1 values	*/
/****************************************************************************/
	int nrows;
/************************************************/
/* generate a sparse matrix from a list of		*/
/* [col, row, element] tri						*/
/************************************************/

	int i, j, jajp1, nza, k, nzrow;
	double xi;

/************************************************/
/* how many rows of result						*/
/************************************************/
	nrows = lastrow - firstrow + 1;

/************************************************/
/* ...count the number of triples in each row	*/
/************************************************/

	for( j=1; j<=n; ++j )
	{
		rowstr[j] = 0;
		mark[j] = false;
	}
	rowstr[n+1] = 0;

	for( nza=1; nza<=nnza; ++nza )
	{
		j = (arow[nza] - firstrow + 1) + 1;
		rowstr[j] = rowstr[j] + 1;
	}
	
	rowstr[1] = 1;
	for( j=2; j<=nrows+1; ++j )
	{
		rowstr[j] = rowstr[j] + rowstr[j-1];
	}
/************************************************************/
/* ... rowstr(j) now is the location of the first nonzero	*/
/* of row j of a											*/
/************************************************************/

/********************************************************/
/* ... do a bucket sort of the triples on the row index	*/
/********************************************************/
	for( nza=1; nza<=nnza; ++nza )
	{
		j = arow[nza] - firstrow + 1;
		k = rowstr[j];
		a[k] = aelt[nza];
		colidx[k] = acol[nza];
		rowstr[j] = rowstr[j] + 1;
	}

/************************************************************/
/* ... rowstr(j) now points to the first element of row j+1	*/
/************************************************************/
	for( j=nrows; j>=1; --j )
	{
		rowstr[j+1] = rowstr[j];
	}
	rowstr[1] = 1;

/************************************************************/
/* ... generate the actual output rows by adding elements	*/
/************************************************************/
	nza = 0;
	for( i=1; i<=n; ++i )
	{
		x[i] = 0.0;
		mark[i] = false;
	}

	jajp1 = rowstr[1];
	for( j=1; j<=nrows; ++j )
	{
		nzrow = 0;

/************************************************************/
/* ...loop over the jth row of a							*/
/************************************************************/
		for( k=jajp1; k<=rowstr[j+1]-1; ++k )
		{
			i = colidx[k];
			x[i] = x[i] + a[k];

			if( (!mark[i]) && (x[i] != 0.0) )
			{
				mark[i] = true;
				nzrow = nzrow + 1;
				nzloc[nzrow] = i;
			}
		}

/************************************************************/
/* ... extract the nonzeros of this row						*/
/************************************************************/
		for( k=1; k<=nzrow; ++k )
		{
			i = nzloc[k];
			mark[i] = false;
			xi = x[i];
			x[i] = 0.0;
			if( xi != 0.0 )
			{
				nza = nza + 1;
				a[nza] = xi;
				colidx[nza] = i;
			}
		}
		jajp1 = rowstr[j+1];
		rowstr[j+1] = nza + rowstr[1];
	}
	printf( "final nonzero count in sparse" );
	printf( " number of nonzeros = %d\n", nza );
}


/******************************************************************/
/*************                M A K E A                ************/
/******************************************************************/
void makea(	int n, int nz, double a[], 
			int colidx[], int rowstr[], int nonzer,
			int firstrow, int lastrow, 
			int firstcol, int lastcol, 
			double rcond,
			double shift )
{
    int i, ivelt, ivelt1;

    double *aelt = harness_malloc_double(nz+1);
    double *v    = harness_malloc_double(n+1+1);
    int *arow    = harness_malloc_int(nz+1);
    int *acol    = harness_malloc_int(nz+1);
    int *nzloc   = harness_malloc_int(n+1);
    bool *mark   = harness_malloc_bool(n+1);

    // nonzer is approximately  sqrt(nnza/n)
    double size = 1.0;
    double ratio = pow(rcond, 1.0/(double) n);
    int nnza = 0;

/*****************************************************/
/*  Initialize iv(n+1 .. 2n) to zero.				 */
/*  Used by sprnvc to mark nonzero positions		 */
/*****************************************************/
    for (i=1; i<=n; i++) mark[i] = 0;
    
    for (i=1; i<=n; i++) {
        int nzv = nonzer;
        sprnvc( n, nzv, v, colidx, nzloc, mark);
        vecset( n, v, colidx, &nzv, i, 0.5 );
        for( ivelt=1; ivelt<=nzv; ++ivelt ) {
            int jcol = colidx[ivelt];
            if( jcol >= firstcol && jcol <= lastcol ) {
                double scale = size * v[ivelt];
                for( ivelt1=1; ivelt1<=nzv; ++ivelt1 ) {
                    int irow = colidx[ivelt1];
                    if( irow >= firstrow && irow <= lastrow ) {
                        nnza = nnza + 1;
                        if (nnza > nz) {
                            printf( "Space for matrix elements exceeded in makea" );
                            printf( "nnza = %d, nzmax = %d", nnza, nz );
                            printf( " iouter = %d\n", i );
                            MPI_Finalize();
                            exit(1);
                        }
                        acol[nnza] = jcol;
                        arow[nnza] = irow;
                        aelt[nnza] = v[ivelt1] * scale;
                    }
                }
            }
        }
        size = size * ratio;
    }




/********************************************************************/
/*  ... add the identity * rcond to the generated matrix to bound	*/
/*  	the smallest eigenvalue from below by rcond		 			*/
/********************************************************************/
		for( i=firstrow; i<=lastrow; ++i )
		{
			if( i >= firstcol && i <= lastcol )
			{
				nnza = nnza + 1;
				if( nnza > nz ) break;
				acol[nnza] = i;
				arow[nnza] = i;
				aelt[nnza] = rcond - shift;
			}
		}

/*********************************************************************/
/*  ... make the sparse matrix from list of elements with duplicates */
/*  	(v and iv are used as  workspace)	 						 */
/*********************************************************************/
		sparse( a, colidx, rowstr, 
				n, arow, acol, aelt, 
				firstrow, lastrow,
				v, nzloc, mark, nnza);

    free(aelt);
    free(v);
    free(acol);
    free(arow);
    free(nzloc);
    free(mark);
}



// process relative data
typedef struct {
    int my_rank;        // process id
    int n;              // benchmark matrix size
    int pcols;         // processes per column
    int prows;         // processes per row
    int exch_proc;      // transpose process

    int firstrow;
    int lastrow;
    int firstcol;
    int lastcol;
    int send_start;
    int send_len;
} submatrix_t;






/******************************************************************/
/************* S E T U P _ S U B M A T R I X _ I N F O ************/
/******************************************************************/
void setup_submatrix(
    submatrix_t *submatrix,
    int my_rank,
    int naa,
    int pcols,
    int prows)
{
    int col_size, row_size;
    int proc_row = my_rank / pcols;
    int proc_col = my_rank - proc_row * pcols;

    submatrix->my_rank  = my_rank;
    submatrix->n        = naa;
    submatrix->pcols   = pcols;
    submatrix->prows   = prows;

/******************************************************************/
/* If naa evenly divisible by pcols, then it is evenly divisible */
/* by prows 													  */
/******************************************************************/
	if( naa/pcols * pcols == naa ) 
	{
		col_size = naa / pcols;
		submatrix->firstcol = proc_col * col_size + 1;
		submatrix->lastcol  = submatrix->firstcol - 1 + col_size;
		row_size = naa / prows;
		submatrix->firstrow = proc_row * row_size + 1;
		submatrix->lastrow  = submatrix->firstrow - 1 + row_size;
	}

/*****************************************************************************/
/* If naa not evenly divisible by pcols, then first subdivide for prows    */
/* and then, if pcols not equal to prows (i.e., not a sq number of procs), */
/* get col subdivisions by dividing by 2 each row subdivision.               */
/*****************************************************************************/
	else
	{
		if( proc_row < naa - naa/prows * prows )
		{
			row_size = naa / prows + 1;
			submatrix->firstrow = proc_row * row_size + 1;
			submatrix->lastrow = submatrix->firstrow - 1 + row_size;
		}
		else
		{
			row_size = naa / prows;
            submatrix->firstrow = (naa - naa/prows * prows) * (row_size + 1)
						+ (proc_row - (naa - naa/prows * prows)) 
						* row_size + 1;
			submatrix->lastrow  = submatrix->firstrow - 1 + row_size;
		}

		if( pcols == prows )
		{
			if( proc_col < naa - naa/pcols * pcols )
			{
				col_size = naa / pcols + 1;
				submatrix->firstcol = proc_col * col_size + 1;
				submatrix->lastcol  = submatrix->firstcol - 1 + col_size;
			}
			else
			{
				col_size = naa / pcols;
				submatrix->firstcol = (naa - naa/pcols * pcols) * (col_size + 1)
							+ (proc_col - (naa - naa/pcols * pcols)) 
							* col_size + 1;
				submatrix->lastcol  = submatrix->firstcol - 1 + col_size;
			}
		}
		else 
		{
			if( proc_col/2 < naa - naa/(pcols/2) * (pcols/2) )
			{
				col_size = naa / (pcols/2) + 1;
				submatrix->firstcol = (proc_col/2) * col_size + 1;
				submatrix->lastcol  = submatrix->firstcol - 1 + col_size;
			}
			else
			{
				col_size = naa / (pcols/2);
				submatrix->firstcol = (naa - naa/(pcols/2)*(pcols/2))
							* (col_size + 1)
							+ ((proc_col/2)-(naa-naa/(pcols/2)*(pcols/2)))
							* col_size + 1;
				submatrix->lastcol  = submatrix->firstcol - 1 + col_size;
			}
			if( my_rank % 2 == 0 )
			{
				submatrix->lastcol = submatrix->firstcol -1 + (col_size - 1)/2 + 1;
			}
			else
			{
				submatrix->firstcol = submatrix->firstcol + (col_size - 1)/2 + 1;
				submatrix->lastcol  = submatrix->firstcol - 1 + col_size/2;
			}
		}
	}

	if( pcols == prows )
	{
		submatrix->send_start = 1;
		submatrix->send_len = submatrix->lastrow  - submatrix->firstrow + 1;
	}
	else
	{
		if( my_rank % 2 == 0 )
		{
			submatrix->send_start = 1;
			submatrix->send_len = (1 + submatrix->lastrow - submatrix->firstrow + 1)/2;
		}
		else
		{
			submatrix->send_start = (1 + submatrix->lastrow - submatrix->firstrow + 1)/2 + 1;
			submatrix->send_len = (submatrix->lastrow - submatrix->firstrow + 1)/2;
		}
	}

/*****************************************************/
/*  Transpose exchange processor					 */
/*****************************************************/
	if( pcols == prows )
	{
		submatrix->exch_proc = (my_rank%prows) * prows + my_rank/prows;
	}
	else
	{
		submatrix->exch_proc = 2 * (((my_rank/2) % prows) * prows + my_rank/2/prows)
					+ my_rank % 2;
	}
}








/******************************************************************/
/*************           C O N J _ G R A D             ************/
/******************************************************************/
double conj_grad(
    submatrix_t *submatrix,
    int colidx[],
    int rowstr[],
    double x[], // input
    double z[], // output
    double a[], // input
    double p[],
    double q[],
    double r[],
    double w[]) // temporary
{
    int j, k;
    double d, sum, rho;
    MPI_Status status;

    int naa        = submatrix->n;
    int pcols      = submatrix->pcols; // could be nprocs as well
    int prows      = submatrix->prows;
    int nrows      = submatrix->lastrow - submatrix->firstrow;
    int ncols      = submatrix->lastcol - submatrix->firstcol;
    int send_start = submatrix->send_start;
    int send_len   = submatrix->send_len;
    int exch_proc  = submatrix->exch_proc;

    for (j=1; j<=naa/prows+1; j++) {
        z[j] = 0.0;
        r[j] = x[j];
        p[j] = r[j];
    }
    // rho = r.r
    sum = 0.0;
    for (j=1; j<=ncols+1; j++) sum = sum + r[j] * r[j];
    MPI_Allreduce(&sum, &rho, 1, MPI_DOUBLE, MPI_SUM, comm_nrow);

    // main conj grad iteration loop
    int cgit;
    for (cgit=0; cgit<CG_ITERATIONS; cgit++) {

        // q = A.p
        // The partition submatrix-vector multiply: use workspace w
        for (j=1; j<=nrows+1; j++) {
            sum = 0.0;
            for (k=rowstr[j]; k<=rowstr[j+1]-1; k++)
                sum = sum + a[k] * p[colidx[k]];
            w[j] = sum;
        }

        // Sum the partition submatrix-vec A.p's across rows
        MPI_Allreduce(&w[1], &q[1], nrows+1, MPI_DOUBLE, MPI_SUM, comm_nrow);

        // Exchange piece of q with transpose processor:
        if (pcols != 1) {
            for (j=1; j<=nrows+1; j++) w[j] = q[j]; // send_start to send_len is enough
            MPI_Sendrecv(&w[send_start], send_len, MPI_DOUBLE, exch_proc, 1,
                &q[1], ncols+1, MPI_DOUBLE, exch_proc, 1,
                comm_nprocs, &status);
        }

        // d = p.q
        sum = 0.0;
        for (j=1; j<=ncols+1; j++) sum = sum + p[j] * q[j];
        MPI_Allreduce(&sum, &d, 1, MPI_DOUBLE, MPI_SUM, comm_nrow);

        // alpha = rho / (p.q)
        // z = z + alpha*p
        // r = r - alpha*q
        double alpha = rho / d;
        for (j=1; j<=ncols+1; j++) {
            z[j] = z[j] + alpha * p[j];
            r[j] = r[j] - alpha * q[j];
        }

        // rho = r.r
        double rho_new;
        sum = 0.0;
        for (j=1; j<=ncols+1; j++) sum = sum + r[j] * r[j];
        MPI_Allreduce(&sum, &rho_new, 1, MPI_DOUBLE, MPI_SUM, comm_nrow);

        // p = r + beta*p
        double beta = rho_new / rho;
        for (j=1; j<=ncols+1; j++) p[j] = r[j] + beta * p[j];
        rho = rho_new;
    }



    // Compute residual norm explicitly:  ||r|| = ||x - A.z||
    // First, form A.z
    // The partition submatrix-vector multiply
    for (j=1; j<=nrows+1; j++) {
        sum = 0.0;
        for (k=rowstr[j]; k<=rowstr[j+1]-1; k++)
            sum = sum + a[k] * z[colidx[k]];
        w[j] = sum;
    }

    // Sum the partition submatrix-vec A.z's across rows
    MPI_Allreduce(&w[1], &r[1], nrows+1, MPI_DOUBLE, MPI_SUM, comm_nrow);

    // Exchange piece of q with transpose processor:
    if (pcols != 1) {
        for (j=1; j<=nrows+1; j++) w[j] = r[j]; // send_start to send_len is enough
        MPI_Sendrecv(&w[send_start], send_len, MPI_DOUBLE, exch_proc, 1,
            &r[1], ncols+1, MPI_DOUBLE, exch_proc, 1,
            comm_nprocs, &status);
    }

    // At this point, r contains A.z
    sum = 0.0;
    for(j=1; j<=ncols+1; j++) {
        d = x[j] - r[j];
        sum = sum + d*d;
    }

    MPI_Allreduce(&sum, &d, 1, MPI_DOUBLE, MPI_SUM, comm_nrow);
    return d;
}




static struct {
    int n;
    int nonzero_per_row;
    int iterations;
    double eigenvalue_shift;
    double verify;
} parameters[7] = {
    {   1400,  7,  15,   10.0, 8.5971775078648}, // class S
    {   7000,  8,  15,   12.0, 10.362595087124}, // class W
    {  14000, 11,  15,   20.0, 17.130235054029}, // class A
    {  75000, 13,  75,   60.0, 22.712745482631}, // class B
    { 150000, 15,  75,  110.0, 28.973605592845}, // class C
    {1500000, 21, 100,  500.0, 52.514532105794}, // class D
    {9000000, 26, 100, 1500.0, 77.522164599383}, // class E
};



/*****************************************************************/
/*************             M  A  I  N             ****************/
/*****************************************************************/
int main(int argc, char **argv)
{
    char class;
    int i, j, k, it;
    int my_rank;
    int comm_size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

/******************************************************************************/
/* num_procs must be a power of 2, and num_procs=num_proc_cols*num_proc_rows. */
/* When num_procs is not square, then num_proc_cols must be = 2*num_proc_rows.*/
/******************************************************************************/
    int log2_num_procs = 8*sizeof(comm_size) - 1 - __builtin_clz(comm_size);
    int num_procs = 1 << log2_num_procs;
    int num_proc_rows = 1 << (log2_num_procs>>1);
    int num_proc_cols = num_procs / num_proc_rows;

    // splitting must be done before MPI_Finalize()
    parop_comm_split_consecutive(MPI_COMM_WORLD, num_procs, &comm_nprocs);
    parop_comm_split_consecutive(MPI_COMM_WORLD, num_proc_cols, &comm_nrow);

    // terminate processes that cannot be used
    if (my_rank >= num_procs) {
        MPI_Finalize();
        exit(0);
    }



    if (argc==2) class=argv[1][0]; else class='S';

    switch (class) {
        case 'S': i = 0; break;
        case 'W': i = 1; break;
        case 'A': i = 2; break;
        case 'B': i = 3; break;
        case 'C': i = 4; break;
        case 'D': i = 5; break;
        case 'E': i = 6; break;
        default:
            printf("Unknown class '%c'\n", class);
            exit(1);
    }
    int na                   = parameters[i].n;
    int nonzer               = parameters[i].nonzero_per_row;
    int niter                = parameters[i].iterations;
    double shift             = parameters[i].eigenvalue_shift;
    double zeta_verify_value = parameters[i].verify;

    if (my_rank == 0) {
        printf("Parallel Benchmark CG\n"
            "  size: %d iterations: %d nonzeros/row: %d eigenvalue shift: %g\n"
            "Using %d of %d processes\n",
            na, niter, nonzer, shift, num_procs, comm_size);
    }

    int nz = na * (nonzer+1) / num_procs * (nonzer+1) + nonzer 
       + na * (nonzer+2 + num_procs / 256) / num_proc_cols;

    int *colidx  = harness_malloc_int(nz+1);
    int *rowstr  = harness_malloc_int(na+1+1);
    double *a    = harness_malloc_double(nz+1);
    double *x    = harness_malloc_double(na/num_proc_rows+2+1);
    double *z    = harness_malloc_double(na/num_proc_rows+2+1);
    double *p    = harness_malloc_double(na/num_proc_rows+2+1);
    double *q    = harness_malloc_double(na/num_proc_rows+2+1);
    double *r    = harness_malloc_double(na/num_proc_rows+2+1);
    double *w    = harness_malloc_double(na/num_proc_rows+2+1);

    // init process relative data
    submatrix_t submatrix;
    setup_submatrix(&submatrix, my_rank, na, num_proc_cols, num_proc_rows);

    randNAS_double(); // only needed for compatibility to FORTRAN version


    // Set up partition's sparse random matrix for given class size
    makea(na, nz, a, colidx, rowstr, nonzer,
        submatrix.firstrow, submatrix.lastrow,
        submatrix.firstcol, submatrix.lastcol,
        0.1, shift);


/*****************************************************************************/
/* Note: as a result of the above call to makea:							 */
/*       values of j used in indexing rowstr go from 1 --> lastrow-firstrow+1*/
/*       values of colidx which are col indexes go from firstcol --> lastcol */
/*       So:																 */
/*       Shift the col index vals from actual (firstcol --> lastcol ) 		 */
/*       to local, i.e., (1 --> lastcol-firstcol+1)							 */
/*****************************************************************************/
    int nrows = submatrix.lastrow - submatrix.firstrow;
	for( j=1; j<=nrows+1; ++j )
	{
		for( k=rowstr[j]; k<=rowstr[j+1]-1; ++k )
		{
			colidx[k] = colidx[k] - submatrix.firstcol + 1;
		}
	}


//  NOTE: a questionable limit on size:  should this be na/num_proc_cols+1 ?
    for( i=1; i<=na/num_proc_rows+1; ++i )
        x[i] = 1.0;

    MPI_Barrier(comm_nprocs);


    // start timer


    // main iteration for inverse power method
    double zeta = 0.0; // avoid warning
    for(it=1; it<=niter; it++) {

        double rsquare = conj_grad(&submatrix, colidx, rowstr, x, z, a, p, q, r, w);

        double local[2], total[2];
        local[0] = 0.0;
        local[1] = 0.0;
        for (j=1; j<=nrows+1; j++) {
            local[0] += x[j] * z[j];
            local[1] += z[j] * z[j];
        }
        MPI_Allreduce(local, total, 2, MPI_DOUBLE, MPI_SUM, comm_nprocs);
        zeta = shift + 1.0 / total[0];

        // normalize z to obtain x
        double norm = 1.0 / sqrt( total[1] );
        for( j=1; j<=nrows+1; ++j )
            x[j] = norm * z[j];

        if (my_rank == 0) {
            if( it == 1 )
                printf( "iteration      ||r||          zeta\n" );
            printf( "%8d%15e%15e\n", it, sqrt(rsquare), zeta);
        }
    }

    // stop timer

    if (my_rank==0) {
        double error = fabs(zeta-zeta_verify_value) / zeta_verify_value;
        if (error <= 1e-10) 
            printf("Completed sucessfully (zeta=%g error=%g)\n", zeta, error);
        else
            printf("VERIFICATION FAILED (zeta=%g error=%g)\n", zeta, error);
    }
    MPI_Finalize();
    return 0;
}
