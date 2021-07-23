#include <stdio.h>
#include <stdlib.h>

#include <SFML/Graphics.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "math.h"
#define PI 3.1415926535897932384626433832795028841971693993751058209749445923078

// THIS WORKS, DON'T TWEAK THIS
#define IMG_INDEX(i,j,k, width, height, channels) ((j)*(width)*(channels)+(i)*(channels)+(k))
// channels should always be 4
#define TEXTURE_INDEX(i,j,k, width, height, channels) ((j)*(width)*(channels)+(i)*(channels)+(k))

#define THREADS_PER_BLOCK 16 // 1 dimensional
#define THREADS(N) dim3((THREADS_PER_BLOCK<N?THREADS_PER_BLOCK:N),1,1);
#define BLOCKS(N) dim3((N>THREADS_PER_BLOCK?N/THREADS_PER_BLOCK:1),1,1);


// #define BENCHMARK
#include "BenchMark.hpp"

// #define DEBUG
// #define COUNT
#include "Debug.hpp"

#include "cuErrorChecking.cuh"
#include "RK-kernels.cuh"

#define DEBUG_ODE

typedef bool (*ODEfunc)(int n_param, double* params, int n_coords, double* coords, double* output);


bool ode_exp(int n_params, float *params, int n_coords, float* coords, float *output){
    if (n_params<1 | n_coords<1 | n_params != n_coords){
        return false;
    }
    for(int i=0; i<n_coords; ++i){
        output[i] = params[i]*coords[i];
    }
    return true;
}

// extern void run_DOPRI5_until(int n_param, float* params, int n_coords, float *coords0, ODEfunc f,
// int MAX_ITER, float* max_coords, int* conv_iter_n, float* coords_iterations, float step_size, uint8_t* flag){

int main(int argc, char** argv){
    // first: test DOPRI5 algorithm

    float param_range[2]{0,100.};
    int n_params = 1024;

    dim3 blocks(gridN/THREADS_PER_BLOCK, gridN/THREADS_PER_BLOCK, 1);
    dim3 threads(THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1);

    // memory on graphics card
    float *params = (float*) malloc(n_params*sizeof(float));
    float *d_params;
    cudaMalloc( (void**) &d_params, n_params*sizeof(float));

    float *coords0 = (float*) malloc(n_params*sizeof(float));
    float *d_coords0;
    cudaMalloc( (void**) &d_coords0, n_params*sizeof(float));

    float *output = (float*) malloc(n_params*sizeof(float));
    float *d_output;
    cudaMalloc( (void**) &d_output, n_params*sizeof(float));

    for(int i=0; i<n_params; ++i){
        params[i] = param_range[0] + (param_range[1]-param_range[0]) * (0.+i)/n_params;
        coords0[i] = 1.0f;
    }

    cudaMemcpy(d_coords0, coords0, n_params*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_params, params, n_params*sizeof(float), cudaMemcpyHostToDevice);

    ODEfunc *ode_exp_f = &ode_exp;
    ODEfunc *d_ode_exp_f;
    cudaMalloc( (void**)d_ode_exp_f )

    run_DOPRI5_until<<<blocks, threads>>>(n_params, d_params, n_params, d_coords0, d_ode_exp_f);





    return 0;
}