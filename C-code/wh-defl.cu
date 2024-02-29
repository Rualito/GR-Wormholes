// Generates a file with the light paths considering some wormhole parameters


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


#define BENCHMARK
#include "BenchMark.hpp"

// #define DEBUG
// #define COUNT
#include "Debug.hpp"

#include "cuErrorChecking.cuh"

// defines the ode functions to use
#include "ODEfunc-wormhole.cuh" 

#include "RK-kernels.cuh"

#define DEBUG_ODE

#define MAX_ITER 2048 // how many iterations until the simulation stops

// extern void run_DOPRI5_until(int n_param, float* params, int n_coords, float *coords0, ODEfunc f,
// int MAX_ITER, float* max_coords, int* conv_iter_n, float* coords_iterations, float step_size, uint8_t* flag){



// end_coords have size n_coords*n_threads; correspond to the final position when simulation stops, either at max_l or MAX_ITER
// void compute_wormhole_deflections(int n_threads, float *coord_array, float*params_array, float max_l, float* end_coords) {
//     const int n_coords = 6; 
//     const int n_params = 5;
    
//     // memory on graphics card
//     float *d_params_array; // device parameters
//     cudaMalloc( (void**) &d_params_array, n_points*n_params*sizeof(float));
//     cudaMemcpy(d_params_array, params_array, n_points*n_params*sizeof(float), cudaMemcpyHostToDevice);
    
//     float *d_coord_array; // device coordinates
//     cudaMalloc( (void**) &d_coord_array, n_points*n_coords*sizeof(float));
//     cudaMemcpy(d_coord_array, coord_array, n_points*n_coords*sizeof(float), cudaMemcpyHostToDevice);

//     float *integration_results = (float*) malloc(n_points*MAX_ITER*n_coords*sizeof(float));
//     float *d_integration_results;
//     cudaMalloc( (void**) &d_integration_results, n_points*MAX_ITER*n_coords*sizeof(float));

//     int *integration_end = (int*) malloc(n_points*sizeof(int));
//     int *d_integration_end;
//     cudaMalloc( (void**) &d_integration_end, n_points*sizeof(int));

//     float *max_coords = new float[n_coords];
//     max_coords[0] = std::numeric_limits<float>::max(); 
//     max_coords[1] = max_l;
//     max_coords[2] = std::numeric_limits<float>::max();
//     max_coords[3] = std::numeric_limits<float>::max();
//     max_coords[4] = std::numeric_limits<float>::max();
//     max_coords[5] = std::numeric_limits<float>::max();
    
//     float *d_max_coords;
//     cudaMalloc((void**) &d_max_coords, n_coords*sizeof(float));
//     cudaMemcpy(d_max_coords, max_coords, n_coords*sizeof(float), cudaMemcpyHostToDevice);
    
//     uint8_t *flag = new uint8_t[1];
//     *flag = 0;
//     uint8_t *d_flag;
//     cudaMalloc((void**) &d_flag, sizeof(uint8_t));
//     cudaMemcpy(d_flag, flag, sizeof(uint8_t), cudaMemcpyHostToDevice);
    
//     // cudaMemcpy(d_coords0, coords0, 2*sizeof(float), cudaMemcpyHostToDevice);
//     // cudaMemcpy(d_params, params, 1*sizeof(float), cudaMemcpyHostToDevice);
//     // a
//     float initial_step_size = 2.0e-1f;
//     float rtol = 1.0e-2f;  
    
//     dim3 blocks = BLOCKS(n_points);
//     dim3 threads = THREADS(n_points);
//     printf("Before CUDA\n");
//     BENCHMARK_START(0);
//     // https://stackoverflow.com/questions/49946929/pass-function-as-parameter-in-cuda-with-static-pointers
//     //extern void run_DOPRI5_coord0_range_until(int n_param, float* params, int n_coords, float *coordsS, float *coordsF, int n_points, ODEfunc f,
//     //   int MAX_ITER, float* max_coords, int* conv_iter_n, float* coords_iterations_range, float step_size, uint8_t* flag) 

//     run_DOPRI5_coord_arr_until<<<blocks, threads>>>(n_params, d_params_array, n_coords, d_coord_array,  
//         MAX_ITER, d_max_coords, d_integration_end, d_integration_results, initial_step_size, rtol, d_flag);
//     gpuErrchk(cudaGetLastError(), false);
    
//     // can do for loop, change d_params and iterate
//     // *params = 2;
//     // cudaMemcpy(d_params, params, 1*sizeof(float), cudaMemcpyHostToDevice);
//     cudaDeviceSynchronize();
    
//     BENCHMARK_END(0);

//     BENCHMARK_START(1);
//     // obtaining the results from the device
//     cudaMemcpy(integration_results, d_integration_results, n_points*MAX_ITER*n_coords*sizeof(float), cudaMemcpyDeviceToHost);
//     cudaMemcpy(integration_end, d_integration_end, n_points*sizeof(int), cudaMemcpyDeviceToHost);
//     cudaMemcpy(flag, d_flag, sizeof(uint8_t), cudaMemcpyDeviceToHost);
//     BENCHMARK_END(1);

//     printf("Result flag: %x\n", *flag);

//     for(int i=0; i<n_points; i++){
//         for(int j=0; j<n_coords; j++){
//             int iteration_end = integration_end[i]-1;
//             end_coords[i*n_coords+k] = integration_results[iteration_end*n_coords+k];
//         }
//     }

//     cudaFree(d_coord_array);
//     cudaFree(d_params_array);
//     cudaFree(d_integration_results);
//     cudaFree(d_integration_end);
//     cudaFree(d_max_coords);
//     cudaFree(d_flag);

//     delete flag;
//     free(integration_results);
//     free(integration_end);

//     // delete[] coord_array;
//     // delete[] params_array;
//     delete[] max_coords;
// }

// int main(int argc, char** argv){
//     // Calculating phi range deflections
//     printf("ODE: %s\n", __cuda_ode_func_name_d__);


//     //                 Initialization of basic parameters
    
//     const int n_points = 128; // n_threads
//     const float max_l = 1000;

//     const float M = 10; 
//     const float a = 5;
//     const float rho = 15;

//     const float l0 = 100;

//     const int n_coords = 6; 
//     const int n_params = 5;

//     //                 Obtaining the initial coordinates to integrate from

//     float *coord_array = new float[n_points*n_coords]; 
//     float *params_array = new float[n_points*n_params];
//     float *end_coords = new float[n_points*n_coords];

//     float t0 = 0;
//     // position in altered spherical coordinates
//     // float l0 = 100;
//     float th0 = PI/2; 
//     float phi0 = 0; // constant   
    
//     for(int i=0; i<n_points; ++i){ // constructing coord_array and params_array to run in the experiments
//         // momentum direction // on the xz plane, along pl and pphi
//         float p = 1; // absolute value of momentum  
//         float p_dir_phi = 0; 
//         float p_dir_th = (PI*i)/n_points; // direction along xz plane varies for multiple experiments

//         float pl0 = p*cos(p_dir_th); //  z axis
//         float pth0 = -p*sin(p_dir_phi)*sin(p_dir_th); // -y axis
//         float pphi0 = p*cos(p_dir_phi)*sin(p_dir_th); // x axis

//         float sth = sin(th0);
//         float B = sqrt(pth0*pth0 + pphi0*pphi0/(sth*sth));  // constant
//         // time, x, y
//         float coords[n_coords]{t0,l0,th0, phi0, pl0, pth0};
//         float params[n_params]{M, a, rho, pphi0, B};

//         for(int j=0; j<n_coords; ++j){
//             coord_array[i*n_coords+j] = coords[j]; 
//         }
//         for(int j=0; j<n_params; ++j){
//             params_array[i*n_params+j] = params[j]; 
//         }
//     }

//     //                         Integrating the ODE and obtaining the results (end_coords)

//     compute_wormhole_deflections(n_points, coord_array, params_array, max_l, end_coords); 




//     delete[] end_coords;
//     delete[] coord_array;
//     delete[] params_array;

//     return 0;
// }
// //*/



// /*

int main(int argc, char** argv){
    // Calculating phi range deflections
    printf("ODE: %s\n", __cuda_ode_func_name_d__);

    const int n_points = 2048; // n_threads
    const int n_coords = 6; 
    const int n_params = 5;

    const float M = 10; 
    const float a = 5;
    const float rho = 15; 

    float *coord_array = new float[n_points*n_coords]; 
    float *params_array = new float[n_points*n_params];

    const float max_l = 500;   

    float t0 = 0;
    // position in altered spherical coordinates
    float l0 = 100;
    float th0 = PI/2; 
    float phi0 = 0; // constant   
    
    for(int i=0; i<n_points; ++i){ // constructing coord_array and params_array to run in the experiments
        // momentum direction // on the xz plane, along pl and pphi
        float p = 1; // absolute value of momentum  
        float p_dir_phi = 0; 
        float p_dir_th = (PI*i)/n_points; // direction along xz plane varies for multiple experiments

        float pl0 = p*cos(p_dir_th); //  z axis
        float pth0 = -p*sin(p_dir_phi)*sin(p_dir_th); // -y axis
        float pphi0 = p*cos(p_dir_phi)*sin(p_dir_th); // x axis

        float sth = sin(th0);
        float B = sqrt(pth0*pth0 + pphi0*pphi0/(sth*sth));  // constant
        // time, x, y
        float coords[n_coords]{t0,l0,th0, phi0, pl0, pth0};
        float params[n_params]{M, a, rho, pphi0, B};

        for(int j=0; j<n_coords; ++j){
            coord_array[i*n_coords+j] = coords[j]; 
        }
        for(int j=0; j<n_params; ++j){
            params_array[i*n_params+j] = params[j]; 
        }
    }
    
    // memory on graphics card
    float *d_params_array; // device parameters
    cudaMalloc( (void**) &d_params_array, n_points*n_params*sizeof(float));
    cudaMemcpy(d_params_array, params_array, n_points*n_params*sizeof(float), cudaMemcpyHostToDevice);
    
    float *d_coord_array; // device coordinates
    cudaMalloc( (void**) &d_coord_array, n_points*n_coords*sizeof(float));
    cudaMemcpy(d_coord_array, coord_array, n_points*n_coords*sizeof(float), cudaMemcpyHostToDevice);

    float *integration_results = (float*) malloc(n_points*MAX_ITER*n_coords*sizeof(float));
    float *d_integration_results;
    cudaMalloc( (void**) &d_integration_results, n_points*MAX_ITER*n_coords*sizeof(float));

    int *integration_end = (int*) malloc(n_points*sizeof(int));
    int *d_integration_end;
    cudaMalloc( (void**) &d_integration_end, n_points*sizeof(int));

    float *max_coords = new float[n_coords];
    max_coords[0] = std::numeric_limits<float>::max(); 
    max_coords[1] = max_l;
    max_coords[2] = std::numeric_limits<float>::max();
    max_coords[3] = std::numeric_limits<float>::max();
    max_coords[4] = std::numeric_limits<float>::max();
    max_coords[5] = std::numeric_limits<float>::max();
    
    float *d_max_coords;
    cudaMalloc((void**) &d_max_coords, n_coords*sizeof(float));
    cudaMemcpy(d_max_coords, max_coords, n_coords*sizeof(float), cudaMemcpyHostToDevice);
    
    uint8_t *flag = new uint8_t[1];
    *flag = 0;
    uint8_t *d_flag;
    cudaMalloc((void**) &d_flag, sizeof(uint8_t));
    cudaMemcpy(d_flag, flag, sizeof(uint8_t), cudaMemcpyHostToDevice);
    
    // cudaMemcpy(d_coords0, coords0, 2*sizeof(float), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_params, params, 1*sizeof(float), cudaMemcpyHostToDevice);
    // a
    float initial_step_size = 2.0e-1f;
    float rtol = 1.0e-2f;  
    
    dim3 blocks = BLOCKS(n_points);
    dim3 threads = THREADS(n_points);
    printf("Before CUDA\n");
    BENCHMARK_START(0);
    // https://stackoverflow.com/questions/49946929/pass-function-as-parameter-in-cuda-with-static-pointers
    //extern void run_DOPRI5_coord0_range_until(int n_param, float* params, int n_coords, float *coordsS, float *coordsF, int n_points, ODEfunc f,
    //   int MAX_ITER, float* max_coords, int* conv_iter_n, float* coords_iterations_range, float step_size, uint8_t* flag) 

    run_DOPRI5_coord_arr_until<<<blocks, threads>>>(n_params, d_params_array, n_coords, d_coord_array,  
        MAX_ITER, d_max_coords, d_integration_end, d_integration_results, initial_step_size, rtol, d_flag);
    gpuErrchk(cudaGetLastError(), false);
    
    // can do for loop, change d_params and iterate
    // *params = 2;
    // cudaMemcpy(d_params, params, 1*sizeof(float), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    
    BENCHMARK_END(0);

    BENCHMARK_START(1);
    // obtaining the results from the device
    cudaMemcpy(integration_results, d_integration_results, n_points*MAX_ITER*n_coords*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(integration_end, d_integration_end, n_points*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(flag, d_flag, sizeof(uint8_t), cudaMemcpyDeviceToHost);
    BENCHMARK_END(1);

    printf("Result flag: %x\n", *flag);
    
    BENCHMARK_START(2);
    std::ofstream outdata;
    outdata.open("integration_results/wormhole_integration_results.dat");
    if(!outdata){
        printf("Can't open file to write results\n");
        exit(1);
    }

    for(int i=0; i<n_points; ++i){ // for each requested point
        outdata << "New point coordinate " << i << "; params:\t";
        for(int j=0; j<n_params; ++j){
            float param_now = params_array[i*n_params+j];
            outdata << param_now << "\t";
        }
        outdata << "\n";
        for(int k=0; k<n_coords; ++k){
            float coord_now = coord_array[i*n_coords+k];
            outdata << coord_now << "\t";
        } 
        outdata <<  "\n";
        //printf("integration end: %d\n", integration_end[i]);
        for(int j=0; j<integration_end[i]; ++j){ // throughout the simulation results
            for(int k=0; k<n_coords; ++k){
                outdata << integration_results[n_coords*MAX_ITER*i + j*n_coords+k] << "\t";
                // 3 coordinates
            }
            outdata << "\n";
        }
        outdata << "\n";
    }
    BENCHMARK_END(2);
        

    cudaFree(d_coord_array);
    cudaFree(d_params_array);
    cudaFree(d_integration_results);
    cudaFree(d_integration_end);
    cudaFree(d_max_coords);
    cudaFree(d_flag);

    delete flag;
    free(integration_results);
    free(integration_end);

    delete[] coord_array;
    delete[] params_array;
    delete[] max_coords;
    
    return 0;
}

//*/