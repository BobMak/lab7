#include <stdio.h>
#define TILEW 32
 
__global__ void matmul_gpu(int *A, int *B, int *C, int d) {
	// Get global thread ID
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int i = idx / d;
    int j = idx % d;
	if ((i < d) && (j < d)) {
        int sum = 0;
        for (int k = 0; k < d; k++)
            sum += A[i*d + k] * B[k*d + j];
        C[i*d + j] = sum;
    }
}

__global__ void matmul_gpu_tiled(int *A, int *B, int *C, int WIDTH) {
    // Get global thread ID
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int row = idx / WIDTH;
    int col = idx % WIDTH;
    int n_tiles = (TILEW + WIDTH -1) / TILEW;

    int TILE_WIDTH = min(WIDTH, TILEW+1);
    int blc_row = blockIdx.x / n_tiles;
    int blc_col = blockIdx.x % n_tiles;
    // int TILE_WIDTH = TILEW;
    int tile_row = threadIdx.x / TILE_WIDTH;
    int tile_col = threadIdx.x % TILE_WIDTH;

    // define shared memory
    __shared__ int As[TILEW * (TILEW+1)];
    __shared__ int Bs[TILEW * (TILEW+1)];
    int Cl = 0;
    int row_idx = (blc_row * TILE_WIDTH + tile_row) * WIDTH;
    int col_idx = tile_col + blc_col*TILE_WIDTH;
    for (int k = 0; k < n_tiles; ++k) { 
        As[tile_row * TILE_WIDTH + tile_col] = A[ row_idx + tile_col + k*TILE_WIDTH ];
        Bs[tile_row * TILE_WIDTH + tile_col] = B[ (k * TILE_WIDTH + tile_row) * WIDTH + col_idx ];
        
        __syncthreads();
        for (int l = 0; l < TILE_WIDTH; ++l)
            Cl += As[tile_row* TILE_WIDTH + l] * Bs[l * TILE_WIDTH + tile_col];
 
        __syncthreads();
    }
    if (row < WIDTH && col < WIDTH)
        C[(tile_row + blc_row*TILE_WIDTH) * WIDTH + blc_col*TILE_WIDTH + tile_col] = Cl;
}

