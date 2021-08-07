#include <stdio.h>
#include <stdlib.h>

#include <SFML/Graphics.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "math.h"
#define PI 3.1415926535897932384626433832795028841971693993751058209749445923078

#include "limits.h"

// THIS WORKS, DON'T TWEAK THIS
#define IMG_INDEX(i,j,k, width, height, channels) ((j)*(width)*(channels)+(i)*(channels)+(k))
// channels should always be 4
#define TEXTURE_INDEX(i,j,k, width, height, channels) ((j)*(width)*(channels)+(i)*(channels)+(k))

#define THREADS_PER_BLOCK 16 // 1 dimensional thread blocks
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

#define MAX_ITER 2048


typedef bool (*ODEfunc)(int n_param, double* params, int n_coords, double* coords, double* output);

__device__
bool ode_test(int n_params, float *params, int n_coords, float* coords, float *output){
    if (n_params<1 | n_coords<2){
        return false;
    }

    output[0] = 1; // time step iteration
    output[1] = coords[2]; //dx/dt = y 
    output[2] = -params[0]*params[0] * coords[1]; //  dy/dt = -w^2 * x
    // these are equations for the harmonic oscillator

    return true;
}

__device__ ODEfunc ode_test_f = ode_test;

// extern void run_DOPRI5_until(int n_param, float* params, int n_coords, float *coords0, ODEfunc f,
// int MAX_ITER, float* max_coords, int* conv_iter_n, float* coords_iterations, float step_size, uint8_t* flag){

int main(int argc, char** argv){
    // first: test DOPRI5 algorithm - harmonic oscillator
    //  dx/dt = y 
    //  dy/dt = -w^2 * x
    // only one parameter w, but many possible starting conditions

    int n_points = 1024;
    // time, x, y
    float pointsS[3]{0,0.,1.};
    float pointsF[3]{0,1.,0.}; 

    // memory on graphics card
    float *params;
    *params = 2*PI;
    float *d_params; // device parameters
    cudaMalloc( (void**) &d_params, 1*sizeof(float));
    cudaMemcpy(d_params, params, 1*sizeof(float), cudaMemcpyHostToDevice);

    float *d_coordsS; // device coordinates
    cudaMalloc( (void**) &d_coordsS, 3*sizeof(float));
    cudaMemcpy(d_coordsS, pointsS, 3*sizeof(float), cudaMemcpyHostToDevice));

    float *d_coordsF; 
    cudaMalloc( (void**) &d_coordsF, 3*sizeof(float));
    cudaMemcpy(d_coordsF, pointsF, 3*sizeof(float), cudaMemcpyHostToDevice));

    float *integration_results = (float*) malloc(n_points*MAX_ITER*3*sizeof(float));
    float *d_integration_results;
    cudaMalloc( (void**) &d_integration_results, n_points*MAX_ITER*3*sizeof(float));

    float *integration_end = (float*) malloc(n_points*sizeof(float));
    float *d_integration_end;
    cudaMalloc( (void**) &d_integration_end, n_points*sizeof(float));


    float *max_coords = (float*) malloc(3*sizeof(float));
    max_coords[0] = 10; // 10 time units of integration
    max_coords[1] = std::numeric_limits<float>::max();
    max_coords[2] = std::numeric_limits<float>::max();
    
    float *d_max_coords;
    cudaMalloc((void**) &d_max_coords, 3*sizeof(float));
    cudaMemcpy(d_max_coords, max_coords, 3*sizeof(float));
    uint8_t *d_flag;
    
    // cudaMemcpy(d_coords0, coords0, 2*sizeof(float), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_params, params, 1*sizeof(float), cudaMemcpyHostToDevice);

    ODEfunc *h_ode_test_f;
    cudaMemcpyFromSymbol(h_ode_test_f, ode_test_f, sizeof(ODEfunc));

    ODEfunc *d_ode_test_f;
    cudaMalloc( (void**)&d_ode_test_f, sizeof(ODEfunc));
    cudaMemcpy(d_ode_test_f, h_ode_test_f, sizeof(ODEfunc), cudaMemcpyHostToDevice);


    float initial_step_size = 1.0e-4f;

    BENCHMARK_START(0);
    // https://stackoverflow.com/questions/49946929/pass-function-as-parameter-in-cuda-with-static-pointers
    //extern void run_DOPRI5_coord0_range_until(int n_param, float* params, int n_coords, float *coordsS, float *coordsF, int n_points, ODEfunc f,
    //   int MAX_ITER, float* max_coords, int* conv_iter_n, float* coords_iterations_range, float step_size, uint8_t* flag) 
    run_DOPRI5_coord0_range_until<<<BLOCKS(n_points), THREADS(n_points)>>>(1, d_params, 2, d_coordsS, d_coordsF, n_points, d_ode_test_f, 
        MAX_ITER, d_max_coords, d_integration_end, d_integration_results, initial_step_size, d_flag);
    
    // can do for loop, change d_params and iterate
    // *params = 2;
    // cudaMemcpy(d_params, params, 1*sizeof(float), cudaMemcpyHostToDevice);

    gpuErrchk(cudaGetLastError(), false);
    cudaDeviceSynchronize();
    
    BENCHMARK_END(0);

    // obtaining the results from the device
    cudaMemcpy(integration_results, d_integration_results, n_points*MAX_ITER*3*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(integration_end, d_integration_end, n_points*sizeof(float), cudaMemcpyDeviceToHost);
    
    //    T O  D O
    // saving the results to a file, to be later intrepreted by some python code (easier to draw)
    

    cudaFree(d_coordsS);
    cudaFree(d_coordsF);
    cudaFree(d_integration_results);
    cudaFree(d_integration_end);
    cudaFree(d_ode_test_f);
    cudaFree(d_max_coords);

    free(integration_results);
    free(integration_end);
    free(max_coords);
    
    return 0;
}