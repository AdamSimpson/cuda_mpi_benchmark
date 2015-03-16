#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <cstring>
#include <assert.h>
#include "cuda_runtime.h"

// Defining _DIRECT will use CUDA enabled MPICH
//#define _DIRECT

// Defining _PINNED will use pinned host memory
//#define _PINNED

int main(int argc, char **argv)
{
    size_t size;
    size_t min_size = 0;
    size_t max_size = 16777216;
    size_t bytes = sizeof(char)*max_size;
    int num_tests = 10000;
    int num_skips = 1000; // 'Primer' iterations
    int num_procs;
    int rank;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if(num_procs != 2)
	exit(EXIT_FAILURE);

    // Allocate Host Memory, if defined allocated pinned memory
    char *h_send, *h_recv;
    #ifdef _PINNED
      cudaHostAlloc(&h_send, bytes, cudaHostAllocDefault);
      cudaHostAlloc(&h_recv, bytes, cudaHostAllocDefault);  
    #else
      h_send = (char*)malloc(bytes);
      h_recv = (char*)malloc(bytes);
    #endif

    // Allocate Device Memory
    char *d_send, *d_recv;
    cudaMalloc((void**)&d_send, bytes);
    cudaMalloc((void**)&d_recv, bytes);

    // Initialize host buffers
    memset(h_send, 0, bytes);
    memset(h_recv, 0, bytes);
    // Initialize device buffers
    cudaMemset(d_send, 's', bytes);
    cudaMemset(d_recv, 'r', bytes);   

    double begin, end;
    MPI_Status stat;
    int i;

    for(size=min_size; size<max_size; size=(size?size*2:1))
    {
        MPI_Barrier(MPI_COMM_WORLD);

        if(rank==0) {

            for(i=0; i<num_tests + num_skips; i++) {
                // Start timer after num_skips 'priming' iterations
                if(i==num_skips) begin = MPI_Wtime();
                #ifdef _DIRECT
		    MPI_Send(d_send, size, MPI_CHAR, 1, 1, MPI_COMM_WORLD);
		    MPI_Recv(d_recv, size, MPI_CHAR, 1, 1, MPI_COMM_WORLD, &stat);
                #else
                    cudaMemcpy(h_send, d_send, size*sizeof(char), cudaMemcpyDeviceToHost);
                    MPI_Send(h_send, size, MPI_CHAR, 1, 1, MPI_COMM_WORLD);
		    MPI_Recv(h_recv, size, MPI_CHAR, 1, 1, MPI_COMM_WORLD, &stat);
                    cudaMemcpy(d_recv, h_recv, size*sizeof(char), cudaMemcpyHostToDevice);
                #endif
            }

            // Stop timer
            end = MPI_Wtime();

            // Print results in microseconds
            if(rank==0)
                printf("%-15llu: %f\n",size*sizeof(char), (end-begin)*1e6/(num_tests*2.0));        

        }
	else if(rank==1) {
            for(i=0; i<num_tests + num_skips; i++) {
                #ifdef _DIRECT
                    MPI_Recv(d_recv, size, MPI_CHAR, 0, 1, MPI_COMM_WORLD, &stat);
                    MPI_Send(d_send, size, MPI_CHAR, 0, 1, MPI_COMM_WORLD);
                #else
                    cudaMemcpy(h_send, d_send, size*sizeof(char), cudaMemcpyDeviceToHost);
                    MPI_Recv(h_recv, size, MPI_CHAR, 0, 1, MPI_COMM_WORLD, &stat); 
                    MPI_Send(h_send, size, MPI_CHAR, 0, 1, MPI_COMM_WORLD);
                    cudaMemcpy(d_recv, h_recv, size*sizeof(char), cudaMemcpyHostToDevice);
                #endif
            }
        }

    }

    // Clean up
    #ifdef _PINNED
        cudaFreeHost(h_send);
        cudaFreeHost(h_recv);
    #else
        free(h_send);
        free(h_recv);
    #endif
    cudaFree(d_send);
    cudaFree(d_recv);

    MPI_Finalize();

    return 0;
}
