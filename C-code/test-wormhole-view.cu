#include <stdio.h>
#include <stdlib.h>

#include <SFML/Graphics.hpp>

// #define STB_IMAGE_IMPLEMENTATION
// #include "stb_image.h"

#include "math.h"
#define PI 3.1415926535897932384626433832795028841971693993751058209749445923078

#include "limits.h"

// TODO: 
// Define Image class (to avoid stb_image.h multiple definition errors)
// example: https://stackoverflow.com/questions/43348798/double-inclusion-and-headers-only-library-stbi-image#54367765


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

#include "Skybox.cuh"
#include "WormholeView.cuh"

#define DEBUG_ODE

#define MAX_ITER 2048 // how many iterations until the simulation stops

// extern void run_DOPRI5_until(int n_param, float* params, int n_coords, float *coords0, ODEfunc f,
// int MAX_ITER, float* max_coords, int* conv_iter_n, float* coords_iterations, float step_size, uint8_t* flag){

int main(int argc, char** argv){
    // Calculating phi range deflections
    printf("ODE: %s\n", __cuda_ode_func_name_d__);

    Skybox horizon1("./horizon/galaxy_horizon.jpg"); 
    Skybox horizon2("./horizon/saturn_horizon.jpg"); 

    int view_width = 640;
    int view_height = 480;

    WormholeView view(horizon1, horizon2, view_width, view_height);

    float l0 = 100;
    float M = 2;
    float rho = 5;
    float a = 0;
    float max_l = 500;

    view.compute_deflection(l0, M, rho, a, max_l);

    uint8_t *texture = (uint8_t*) malloc(view_width*view_height*horizon1.__channels*sizeof(uint8_t));

    float phi = 0;
    float th = PI/2;

    if(view.get_texture_array(texture, th, phi)){
        printf("Texture array obtained\n");
        stbi_write_jpg("window view.jpg", view_width, view_height, channels, texture, 100);
    } else { 
        printf("Something didn't work\n");
    }

    


    free(texture);

    return 0;
}
//*/
