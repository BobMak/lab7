#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "kernel.cu"

#define DEFAULT_MEM_MODE 0
#define PINNED_MEM_MODE 1
#define UVM_MEM_MODE 2
#define UVM_ACCBY_MEM_MODE 3
#define UVM_PREFLOC_MEM_MODE 4


void init(int *A, int *B, int *C, int d1){
    int i, j;
    // use small maximum ranges to avoid overflow in bigger dimensions
    // initialize A
    for (i=0; i<d1; i++) {
        for (j=0; j<d1; j++) {
            A[i*d1 +j] = (rand() % 256);
        };
    };
    // initialize B
    for (i=0; i<d1; i++) {
        for (j=0; j<d1; j++) {
            B[i*d1 + j] = (rand() % 256);
        };
    };
    // initialize C
    for (i=0; i<d1; i++) {
        for (j=0; j<d1; j++) {
            C[i*d1 + j] = 0;
        };
    };
}

void matmul_cpu(int *A, int *B, int *C, int d1) {
    int i, j, k;
    for (i=0; i<d1; i++) {
        for (j=0; j<d1; j++) {
            int sum = 0;
            for (k=0; k<d1; k++) {
                sum += A[i*d1 + k] * B[k*d1 + j];
            };
            C[i*d1 + j] = sum;
        };
    };
}

void print_mat(int *A, int d1) {
    int i, j;
    for (i=0; i<d1; i++) {
        for (j=0; j<d1; j++) {
            printf("%d\t", A[i*d1 + j]);
        };
        printf("\n");
    };
}

int main(int argc, char *argv[]) {
    int *A, *B, *C;
    int *A_d, *B_d, *C_d;
    int i, j;
    srand(time(NULL));
    // printf("Matrix Multiplication\n");
    // default values
    dim3 blockSize(TILEW*TILEW);
    // read arguments
    if (argc<5){
        printf("Usage: %s <matrix dimensions> <optimization level> <verify result> <memory mode>\n", argv[0]);
        return 1;
    }
    u_int d1 = atoi(argv[1]);
    u_int optimize = atoi(argv[2]);
    if (optimize > 1) {
        printf("Invalid optimization level, should be in (0 - no optimization | 1 - tiled optimization)\n");
        return 1;   
    }
    bool verify_result = atoi(argv[3]) > 0;
    u_int mem_mode = atoi(argv[4]);
    if (mem_mode > 4) {
        printf("Invalid memory mode, should be in (0 - default |1 - pinned |2 - UVM|3 - UVM + accessed by hint| 4 - UVM + preferred location hint)\n");
        return 1;
    }

    int gsize = ceil((float)(d1*d1) / (TILEW*TILEW));
    printf("block size: %d, grid size: %d\n", TILEW*TILEW, gsize);
    dim3 gridSize(gsize);
    // allocate memory
    A = (int*) malloc(d1*d1*sizeof(int*));
    B = (int*) malloc(d1*d1*sizeof(int*));
    C = (int*) malloc(d1*d1*sizeof(int*));
    
    if (mem_mode == DEFAULT_MEM_MODE) {
        cudaMalloc((void **) &A_d, d1*d1*sizeof(int*) );
        cudaMalloc((void **) &B_d, d1*d1*sizeof(int*) );
        cudaMalloc((void **) &C_d, d1*d1*sizeof(int*) );
    }
    else if (mem_mode == PINNED_MEM_MODE) {
        cudaMallocHost((void **) &A_d, d1*d1*sizeof(int*), cudaHostAllocDefault );
        cudaMallocHost((void **) &B_d, d1*d1*sizeof(int*), cudaHostAllocDefault );
        cudaMallocHost((void **) &C_d, d1*d1*sizeof(int*), cudaHostAllocDefault );
    }
    else if (mem_mode >= UVM_MEM_MODE) {
        cudaMallocManaged((void **) &A_d, d1*d1*sizeof(int*) );
        cudaMallocManaged((void **) &B_d, d1*d1*sizeof(int*) );
        cudaMallocManaged((void **) &C_d, d1*d1*sizeof(int*) );
        if (mem_mode == UVM_ACCBY_MEM_MODE) {
            cudaMemAdvise(A_d, d1*d1*sizeof(int*), cudaMemAdviseSetAccessedBy, 0);
            cudaMemAdvise(B_d, d1*d1*sizeof(int*), cudaMemAdviseSetAccessedBy, 0);
            cudaMemAdvise(C_d, d1*d1*sizeof(int*), cudaMemAdviseSetAccessedBy, 0);
        } else if (mem_mode == UVM_PREFLOC_MEM_MODE) {
            cudaMemAdvise(A_d, d1*d1*sizeof(long long int), cudaMemAdviseSetPreferredLocation, 0);
            cudaMemAdvise(B_d, d1*d1*sizeof(long long int), cudaMemAdviseSetPreferredLocation, 0);
            cudaMemAdvise(C_d, d1*d1*sizeof(long long int), cudaMemAdviseSetPreferredLocation, 0);
        } 
    }  
    
    // initialize matrices
    init(A, B, C, d1);
    // move matrices to cuda
    cudaMemcpy(A_d, A, d1*d1*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, d1*d1*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(C_d, C, d1*d1*sizeof(int), cudaMemcpyHostToDevice);
    // synch devices
    cudaDeviceSynchronize();

    // multiply
    if (optimize==0)
        matmul_gpu<<<gridSize, blockSize>>>(A_d, B_d, C_d, d1);
    else if (optimize==1)
        matmul_gpu_tiled<<<gridSize, blockSize>>>(A_d, B_d, C_d, d1);

    // copy results back
    cudaMemcpy(C, C_d, d1*d1*sizeof(int), cudaMemcpyDeviceToHost);
    // synch devices
    cudaDeviceSynchronize();
    // verify with CPU matmul
    if (verify_result) {
        int *C_cpu;
        C_cpu = (int*) malloc(d1*d1*sizeof(int*));
        matmul_cpu(A, B, C_cpu, d1);
        bool pass = true;
        for (i=0; i<d1; i++) {
            for (j=0; j<d1; j++) {
                if (C[i*d1 + j] != C_cpu[i*d1 + j]) {
                    pass = false;
                    printf("Verification failed at %d %d: %d != %d\n", i, j, C[i*d1 + j], C_cpu[i*d1 + j]);
                    break;
                };
            };
        };
        if (pass)
            printf("Verification Success!\n");
        else
            printf("Verification Failed!\n");
        free(C_cpu);
    }
    // free memory
    free(A);
    free(B);
    free(C);
    if (mem_mode == PINNED_MEM_MODE) {
        cudaFreeHost(A_d);
        cudaFreeHost(B_d);
        cudaFreeHost(C_d);
    }
    else {
        cudaFree(A_d);
        cudaFree(B_d);
        cudaFree(C_d);
    }
    return 0;
}