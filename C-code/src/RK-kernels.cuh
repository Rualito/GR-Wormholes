#ifndef RK_KERNELS_H
#define  RK_KERNELS_H

#define THREADS_PER_BLOCK 16 // 1 dimentional
#define THREADS(N) dim3((THREADS_PER_BLOCK<N?THREADS_PER_BLOCK:N),1,1);
#define BLOCKS(N) dim3((N>THREADS_PER_BLOCK?N/THREADS_PER_BLOCK:1),1,1);

//#define BENCHMARK
#include "BenchMark.hpp"

#define COUNT
#include "Debug.hpp"

#include "cuErrorChecking.cuh"

#include "math.h"

// #define DEBUG_CUDA

// DOPRI5 constants
// https://en.wikipedia.org/wiki/Dormand%E2%80%93Prince_method
__constant__ float b_factor[7]{5179./57600, 0, 7571./16695, 393./640, -92097./339200, 187./2100, 1./40};
__constant__ float b_star_factor[7]{35./384, 0, 500./1113, 125./192, -2187./6784, 11./84, 0};

__constant__ float a2_factor = 1./5;
__constant__ float a3_factor[2]{3./40, 9./40};
__constant__ float a4_factor[3]{44./45, 56./15, 32./9};
__constant__ float a5_factor[4]{19372./6561, -25360./2187, 64448./6561, -212./729};
__constant__ float a6_factor[5]{9017./3168, -355./33, 46732./5247, 49./176, -5103./18656};
__constant__ float a7_factor[6]{35./384, 0, 500./1113, 125./192, -2187./6784, 11./84};

__constant__ float c_factor[6]{1./5, 3./10, 4./5, 8./9, 1., 1.};



// n_param: physical parameters (n_param-pointer)
// coords0: initial conditions (n_coords-pointer)
// coords1: final conditions after 1 step (n_coords-pointer)
// step_size: runge-kutta step, that gets updated after 1 iteration (adaptative) (1-pointer)
// rtol: error tolerance
// adapting step size: https://aip.scitation.org/doi/pdf/10.1063/1.4887558 expr. (4)
// butcher tableau: https://en.wikipedia.org/wiki/Dormand%E2%80%93Prince_method
// __device__
// void rk45_step(int n_param, float* params, int n_coords, float *coords0, ODEfunc f, float *step_size, float rtol, float* coords1){
//     // // TO BE CONTINUED ......
//     // adaptative step
//     // order 4-5, runge-kutta-fehlberg 
//     // check scipy.integrate.rk45 implementation

//     float local_error = rtol*1.1;
    
//     while(local_error>rtol){
        




//         step_size *= ....;
//     }
// }



 // MAX_ITER: integrate until MAX_ITER rk steps have passed 
// max_coords: conditions to run rk steps until coordinates reach absolte values (so either positive or negative)
//  bigger than the respective values (n_coords-pointer)
// conv_iter_n: if stop conditions applied (of max_coords), then is the number of iterations until it converged (1-pointer)
// coords_iterations: array saving all the iteration steps until either convergence or MAX_ITER (MAX_ITER*n_coords-pointer)
// step_size: initial step size 
// rtol: error tolerance
// set max_coords[:] = float_max, then no limit is taken 
// coords[0] is time
// flag changed to 1 if ODE function error 

__device__
void run_DOPRI5_until(int n_params, float* params, int n_coords, float *coords0, int MAX_ITER, float* max_coords, int* conv_iter_n, float* coords_iterations, float step_size, float rtol,uint8_t* flag){
    // const int index_x = blockIdx.x*blockDim.x+threadIdx.x;

    (*conv_iter_n) = 0;

    // if flag is not zero, then abort
    if(*flag) {
        printf("Flag abort: %d\n", *flag);
        return;
    }
    
    float *coords_now = (float*) malloc(n_coords*sizeof(float));
    float *coords_temp = (float*) malloc(n_coords*sizeof(float));

    float *coords_after = (float*) malloc(n_coords*sizeof(float));
    
    // float *coords_after1 = (float*) malloc(n_coords*sizeof(float));
    //float *coords_star = (float*) malloc(n_coords*sizeof(float));
    // initial coordinates
    for(int k=0; k<n_coords; ++k){
        coords_now[k] = coords0[k];
    }
    
    
    bool stop_flag = false;
    // float step = step_size;
    bool success_flag = true;
    // Runge-Kutta coefficients initialization
    float *k1 = (float*) malloc(n_coords*sizeof(float)); 
    float *k2 = (float*) malloc(n_coords*sizeof(float));
    float *k3 = (float*) malloc(n_coords*sizeof(float));
    float *k4 = (float*) malloc(n_coords*sizeof(float));
    float *k5 = (float*) malloc(n_coords*sizeof(float));
    float *k6 = (float*) malloc(n_coords*sizeof(float));
    float *k7 = (float*) malloc(n_coords*sizeof(float));

    *flag = 0; // no error
    // printf("rtol %f\n", rtol);
    // f(n_params, params, n_coords, coords_now,k1);
    // f(n_params, params, n_coords, coords_now,k7);
    int i=0;
    // printf("MAX_ITER : %d\n", MAX_ITER);
    for(i=0; i<MAX_ITER; ++i){
        // rk45_step(n_param, params, n_coords, coords_now, f, &step, rtol, coords_after);
        // printf("Current iteration: %d\n", i);

        // ADAPTATIVE STEP ALGORITHM
        float local_error = rtol*1.1;
        // this isnt valid if step_size changes
        // for(int j=0; j<n_coords; ++j){
        //     k1[j] = k7[j];
        // }

        float error2 = 0;
        while(local_error>rtol){
            // printf("Inside while loop\n");
            // calculate k1
            success_flag &= __cuda_ode_func__(n_params, params, n_coords, coords_now,k1);
            // printf("After k1\n");
            // k2
            coords_temp[0] = coords_now[0] + c_factor[0]*step_size;
            for(int j=1; j<n_coords; ++j){
                coords_temp[j] = coords_now[j] + step_size*a2_factor*k1[j];
            }
            success_flag &= __cuda_ode_func__(n_params, params, n_coords, coords_temp, k2);

            // calculate k3
            coords_temp[0] = coords_now[0] + c_factor[1]*step_size;
            for(int j=1; j<n_coords; ++j){
                coords_temp[j] = coords_now[j] + step_size*(a3_factor[0]*k1[j]+a3_factor[1]*k2[j]);
            }
            success_flag &= __cuda_ode_func__(n_params, params, n_coords, coords_temp, k3);

            // calculate k4
            coords_temp[0] = coords_now[0] + c_factor[2]*step_size;
            for(int j=1; j<n_coords; ++j){
                coords_temp[j] = coords_now[j] + step_size*(a4_factor[0]*k1[j]+a4_factor[1]*k2[j]+a4_factor[2]*k3[j]);
            }
            success_flag &= __cuda_ode_func__(n_params, params, n_coords, coords_temp, k4);
            
            // calculate k5
            coords_temp[0] = coords_now[0] + c_factor[3]*step_size;
            for(int j=1; j<n_coords; ++j){
                coords_temp[j] = coords_now[j] + step_size*(a5_factor[0]*k1[j]+a5_factor[1]*k2[j]+a5_factor[2]*k3[j]+a5_factor[3]*k4[j]);
            }
            success_flag &= __cuda_ode_func__(n_params, params, n_coords, coords_temp, k5);

            // calculate k6
            coords_temp[0] = coords_now[0] + c_factor[4]*step_size;
            for(int j=1; j<n_coords; ++j){
                coords_temp[j] = coords_now[j] + step_size*(a6_factor[0]*k1[j]+a6_factor[1]*k2[j]+a6_factor[2]*k3[j]+a6_factor[3]*k4[j]+a6_factor[4]*k5[j]);
            }
            success_flag &= __cuda_ode_func__(n_params, params, n_coords, coords_temp, k6);

            // calculate k7
            coords_temp[0] = coords_now[0] + c_factor[5]*step_size;
            for(int j=1; j<n_coords; ++j){
                coords_temp[j] = coords_now[j] + step_size*(a7_factor[0]*k1[j]+a7_factor[1]*k2[j]+a7_factor[2]*k3[j]+a7_factor[3]*k4[j]+a7_factor[4]*k5[j]+a7_factor[5]*k6[j]);
            }
            success_flag &= __cuda_ode_func__(n_params, params, n_coords, coords_temp, k7);
            // printf("After k7\n");

            if(!success_flag){ 

                printf("ODE func error\n");
                *flag = 1; // if ODE func error

                return;
            }

            // calculate coords_after and estimate error
            error2 = 0;
            coords_after[0] = coords_now[0] + step_size;
            // printf("Before error loop\n");
            for(int j=1; j<n_coords; ++j){
                // printf("On error loop %d\n", j);
                coords_after[j] = coords_now[j] + step_size*(b_factor[0]*k1[j]+b_factor[1]*k2[j]+b_factor[2]*k3[j]+b_factor[3]*k4[j]+b_factor[4]*k5[j]+b_factor[5]*k6[j]+b_factor[6]*k7[j]);
                float coords_star = coords_now[j] + step_size*(b_star_factor[0]*k1[j]+b_star_factor[1]*k2[j]+b_star_factor[2]*k3[j]+b_star_factor[3]*k4[j]+b_star_factor[4]*k5[j]+b_star_factor[5]*k6[j]);
                error2 += (coords_after[j]-coords_star)*(coords_after[j]-coords_star);
            }
            // printf("After error loop, error2: %e\n", error2);
            local_error = sqrt(error2);
            
            // printf("rtol/local_error: %f\n", rtol/local_error);
            if(local_error > 0){
                step_size *= 0.84*pow((rtol/local_error), 0.2);
            }
            // printf("")
            // printf("step_size: %f\n", step_size);
            // if(isnan(step_size)){ // then fuckery is afoot
            //     printf("coords_after: \n");
            //     for(int j=0; j<n_coords; ++j){
            //         printf("%f ", coords_after[j]);
            //     }
            //     printf("\n");
            // }
            if(isnan(step_size)){
                printf("Somehow step_size is nan, aborting;\n");
                *flag = 2;
                return;
            }
            // if(coords_after[0] > 10 ){
            //     printf("Coords big: %f, local_error: %f\n", coords_after[0], local_error);
            // }
        }

        // printf("After while loop\n");

        for(int k=0; k<n_coords; ++k){
            coords_iterations[i*n_coords+k] = coords_after[k]; // saving results to array
            coords_now[k] = coords_after[k]; // changing the current coords to be new ones after the rk step
            
            stop_flag |=  (fabs(coords_after[k])>max_coords[k]); 
            // if stop_flag is true, then remains always true
            // avoids if conditional
        }
        // printf("After for loop, before stop flag\n");
        if(stop_flag){
            // printf("stop_flag triggered! local error: %f, step_size %f, error2 %f\n", local_error, step_size, error2);
            // printf("Testprint: %e\n", sqrt(1.0e-14f));
            (*conv_iter_n) = i+1;
            break; 
        }
    }
    if(!stop_flag){
        (*conv_iter_n) = i;
    }    

    free(coords_now);
    free(coords_temp);
    free(coords_after);
    // free(coords_after1);
    //free(coords_star);
    free(k1);
    free(k2);
    free(k3);
    free(k4);
    free(k5);
    free(k6);
    free(k7);
    
}


    
// coordsS: coordinate start
// coordsF: coordinate end
// n: how many points to run between the coordinate interval
// if n=0, it is only considered coordsS
// absolute thread id is used to get position within coordinate grid
// coords_iterations_range: array with size n_points*MAX_ITER*n_coords, ordered sequentially; 
// index - coord_index  + coords_n*iteration_index + coords_n*MAX_ITER*point_index  
// ^ such that coordinate components are grouped together, then iteration results, and finally diferent points of simulations  
__global__ 
void run_DOPRI5_coord0_range_until(int n_param, float* params, int n_coords, float *coordsS, float *coordsF, int n_points,
int MAX_ITER, float* max_coords, int* conv_iter_n, float* coords_iterations_range, float step_size, float rtol, uint8_t* flag) {

    const int index_x = blockIdx.x*blockDim.x+threadIdx.x;

    float *coords0 = (float*) malloc(n_coords*sizeof(float));

    // compute the starting coordinate of current thread
    for(int i=0; i<n_coords; ++i){
        coords0[i] = coordsS[i] + ((coordsF[i]-coordsS[i]) * index_x)/n_points;
        // printf("ci: %f, index: %d\n",coords0[i], index_x);
    }
    // printf("Before kernel %d\n", index_x);
    run_DOPRI5_until(n_param, params, n_coords, coords0, MAX_ITER, max_coords, 
        &conv_iter_n[index_x] ,&coords_iterations_range[n_coords*MAX_ITER*index_x], step_size, rtol, flag);

    // printf("After kernel %d\n", index_x);
    
}

// coords_arr : n_coords*n_points array of coordinates to run the simulation [ coord1, coord2, ...] (no seg fault safety, n_points > n_threads)
// n: how many points to run between the coordinate interval
// if n=0, it is only considered coordsS
// absolute thread id is used to get position within coordinate grid
// coords_iterations_range: array with size n_points*MAX_ITER*n_coords, ordered sequentially; 
// index - coord_index  + coords_n*iteration_index + coords_n*MAX_ITER*point_index  
// ^ such that coordinate components are grouped together, then iteration results, and finally diferent points of simulations 
__global__ 
void run_DOPRI5_coord_arr_until(int n_param, float* params_arr, int n_coords, float *coords_arr,
    int MAX_ITER, float* max_coords, int* conv_iter_n, float* coords_iterations_range, float step_size, float rtol, uint8_t* flag) {

    const int index_x = blockIdx.x*blockDim.x+threadIdx.x;
    
    run_DOPRI5_until(n_param, &params_arr[index_x*n_param], n_coords, &coords_arr[index_x*n_coords], MAX_ITER, max_coords, 
        &conv_iter_n[index_x] ,&coords_iterations_range[n_coords*MAX_ITER*index_x], step_size, rtol, flag);
}


#endif