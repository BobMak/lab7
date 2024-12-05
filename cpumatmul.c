#include <stdio.h>
#include <stdlib.h>
#include <time.h>


int **alloc_mat(int d1){
    int i, j;
    // allocate
    int **A = malloc(d1*sizeof(int*));
    for (i=0; i<d1; i++) {
        A[i] = malloc(d1*sizeof(int));
    };
    return A;
}

void init(int **A, int **B, int **C, int d1){
    int i, j;
    // initialize A
    for (i=0; i<d1; i++) {
        for (j=0; j<d1; j++) {
            A[i][j] = (rand() % 64 );// 32768);
        };
    };
    // initialize B
    for (i=0; i<d1; i++) {
        for (j=0; j<d1; j++) {
            B[i][j] = (rand() % 64 );// 32768);
        };
    };
    // initialize C
    for (i=0; i<d1; i++) {
        for (j=0; j<d1; j++) {
            C[i][j] = 0.0;
        };
    };
}

void dealloc(int **A, int d1) {
    int i;
    for (i=0; i<d1; i++) {
        free(A[i]);
    };  
    free(A);
}

void matmul(int **A, int **B, int **C, int n, int m, int p) {
    int i, j, k;
    for (i=0; i<n; i++)
        for (j=0; j<m; j++)
            for (k=0; k<p; k++)
                C[i][j] = C[i][j] + A[i][k] * B[k][j];
}

void matmul_optimized(int **A, int **B, int **C, int n, int m, int p) {
    int ii, jj, kk, i, j, k;
    int b = 4;
    // better row order for cache locality
    for (i=0; i<n; i++)
        for (j=0; j<m; j++)
            for (k=0; k<p; k++)
                C[j][k] = C[j][k] + A[i][j] * B[j][k];
}


void print_mat(int **A, int d1) {
    int i, j;
    for (i=0; i<d1; i++) {
        for (j=0; j<d1; j++) {
            printf("%d ", A[i][j]);
        };
        printf("\n");
    };
}

int main(int argc, char *argv[]) {
    int **A, **B, **C;
    int d1, i, j, optimize;
    srand(time(NULL));
    // default values
    d1 = 3;
    // read arguments
    if (argc<3) {
        printf("Usage: cpumatmul <d1> <optimize>\n");
        return 1;
    };
    d1 = atoi(argv[1]);
    optimize = atoi(argv[2]) > 0;
    // allocate memory
    A = alloc_mat(d1);
    B = alloc_mat(d1);
    C = alloc_mat(d1);
    init(A, B, C, d1);
    // multiply
    if (optimize==0)
        matmul(A, B, C, d1, d1, d1);
    else
        matmul_optimized(A, B, C, d1, d1, d1);
    // print results
    // printf("A:\n");
    // print_mat(A, d1, d1);
    // printf("B:\n");
    // print_mat(B, d1, d1);
    // printf("C:\n");
    // print_mat(C, d1, d1);
    // free memory
    dealloc(A, d1);
    dealloc(B, d1);
    dealloc(C, d1);
    return 0;
}