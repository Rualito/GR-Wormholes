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

#include <iostream>
#include <fstream>

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


typedef bool (*ODEfunc)(int n_param, float* params, int n_coords, float* coords, float* output);

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
    printf("Beginning\n");

    // memory on graphics card
    float *params = new float[1];
    *params = 2*PI;
    float *d_params; // device parameters
    cudaMalloc( (void**) &d_params, 1*sizeof(float));
    cudaMemcpy(d_params, params, 1*sizeof(float), cudaMemcpyHostToDevice);
    printf("params done\n");
    float *d_coordsS; // device coordinates
    cudaMalloc( (void**) &d_coordsS, 3*sizeof(float));
    cudaMemcpy(d_coordsS, pointsS, 3*sizeof(float), cudaMemcpyHostToDevice);

    float *d_coordsF; 
    cudaMalloc( (void**) &d_coordsF, 3*sizeof(float));
    cudaMemcpy(d_coordsF, pointsF, 3*sizeof(float), cudaMemcpyHostToDevice);
    printf("cordsF done\n");

    float *integration_results = (float*) malloc(n_points*MAX_ITER*3*sizeof(float));
    float *d_integration_results;
    cudaMalloc( (void**) &d_integration_results, n_points*MAX_ITER*3*sizeof(float));

    float *integration_end = (float*) malloc(n_points*sizeof(float));
    int *d_integration_end;
    cudaMalloc( (void**) &d_integration_end, n_points*sizeof(int));
    printf("integration_... done\n");

    float *max_coords = (float*) malloc(3*sizeof(float));
    max_coords[0] = 10; // 10 time units of integration
    max_coords[1] = std::numeric_limits<float>::max();
    max_coords[2] = std::numeric_limits<float>::max();
    
    float *d_max_coords;
    cudaMalloc((void**) &d_max_coords, 3*sizeof(float));
    cudaMemcpy(d_max_coords, max_coords, 3*sizeof(float), cudaMemcpyHostToDevice);
    printf("max_coords done\n");
    uint8_t *flag = new uint8_t[1];
    *flag = 0;
    uint8_t *d_flag;
    cudaMalloc((void**) &d_flag, sizeof(uint8_t));
    cudaMemcpy(d_flag, flag, sizeof(uint8_t), cudaMemcpyHostToDevice);

    // cudaMemcpy(d_coords0, coords0, 2*sizeof(float), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_params, params, 1*sizeof(float), cudaMemcpyHostToDevice);

    ODEfunc *h_ode_test_f;
    cudaMemcpyFromSymbol(h_ode_test_f, ode_test_f, sizeof(ODEfunc));

    ODEfunc *d_ode_test_f;
    cudaMalloc( (void**)&d_ode_test_f, sizeof(ODEfunc));
    cudaMemcpy(d_ode_test_f, h_ode_test_f, sizeof(ODEfunc), cudaMemcpyHostToDevice);


    float initial_step_size = 1.0e-4f;
    float rtol = 1.0e-6;
    
    dim3 blocks = BLOCKS(n_points);
    dim3 threads = THREADS(n_points);
    printf("Before CUDA\n");
    BENCHMARK_START(0);
    // https://stackoverflow.com/questions/49946929/pass-function-as-parameter-in-cuda-with-static-pointers
    //extern void run_DOPRI5_coord0_range_until(int n_param, float* params, int n_coords, float *coordsS, float *coordsF, int n_points, ODEfunc f,
    //   int MAX_ITER, float* max_coords, int* conv_iter_n, float* coords_iterations_range, float step_size, uint8_t* flag) 
    run_DOPRI5_coord0_range_until<<<blocks, threads>>>(1, d_params, 3, d_coordsS, d_coordsF, n_points, d_ode_test_f, 
        MAX_ITER, d_max_coords, d_integration_end, d_integration_results, initial_step_size, rtol,d_flag);
    
    // can do for loop, change d_params and iterate
    // *params = 2;
    // cudaMemcpy(d_params, params, 1*sizeof(float), cudaMemcpyHostToDevice);

    gpuErrchk(cudaGetLastError(), false);
    cudaDeviceSynchronize();
    
    BENCHMARK_END(0);

    // obtaining the results from the device
    cudaMemcpy(integration_results, d_integration_results, n_points*MAX_ITER*3*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(integration_end, d_integration_end, n_points*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(flag, d_flag, sizeof(uint8_t), cudaMemcpyDeviceToHost);
    
    printf("After CUDA\n");

    printf("Result flag: %d\n", flag);

    std::ofstream outdata;
    outdata.open("integration_results.dat");
    if(!outdata){
        printf("Can't open file to write results\n");
        exit(1);
    }
    float coord0[3];
    for(int i=0; i<n_points; ++i){ // for each requested point
        outdata << "New point coordinate " << i << "\n";
        for(int k=0; k<3; ++k){
            float coord_now = pointsS[k] + ((pointsF[k]-pointsF[k]) * i*1.0) / n_points;
            outdata << coord_now << " ";
        } 
        outdata <<  "\n";

        for(int j=0; j<integration_end[i]; ++j){ // throughout the simulation results
            for(int k=0; k<3; ++k){
                outdata << integration_results[3*MAX_ITER*i + j*3+k] << " ";
                // 3 coordinates
            }
            outdata << "\n";
        }
        outdata << "\n";
    }

    //    T O  D O
    // saving the results to a file, to be later intrepreted by some python code (easier to draw)
        

    cudaFree(d_coordsS);
    cudaFree(d_coordsF);
    cudaFree(d_integration_results);
    cudaFree(d_integration_end);
    cudaFree(d_ode_test_f);
    cudaFree(d_max_coords);
    cudaFree(d_flag);

    delete params;
    delete flag;
    free(integration_results);
    free(integration_end);
    free(max_coords);
    
    return 0;
}