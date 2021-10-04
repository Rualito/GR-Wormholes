


// An ODE func ( f(n_param, *params, n_coords, *coords, *output) )
// that takes in physical parameters (like mass, radii, etc) 
// and coordinate points (could be position and/or momentum)
// in any coordinate system
// output has size n_coords
// returns bool : true if successful 

// typedef bool (*ODEfunc)(int n_param, float* params, int n_coords, float* coords, float* output);

#ifndef __cuda_ode_func__

#define PI 3.1415926535897932384626433832795028841971693993751058209749445923078
#include "math.h"

// Wormhole differential equation
__device__
bool __cuda_ode_func__(int n_params, float *params, int n_coords, float* coords, float *output){
    if (n_params<5 | n_coords<6){
        return false;
    }
    
    // coords: 
    // [0] : t - time
    // [1] : l - distance through wormhole
    // [2] : th - theta, altitude angle
    // [3] : phi - azimuth angle
    // [4] : pl - radial direction of light momentum
    // [5] : pth - polar direction of light momentum  

    // params: 
    // [0] : M - wormhole mass
    // [1] : a - wormhole throat length
    // [2] : rho - wormhole throat radius
    // [3] : pphi - azimuthal direction of light momentum, constant (implemented this way to be variable through threads)
    // [4] : B - impact parameter relative to center, also constant, same reason and avoiding to calculate sqrt each time

    float l = coords[1];
    float th = coords[2];
    // float phi = coords[3];
    float pl = coords[4];
    float pth = coords[5];
    

    float sth = sin(th);
    float cth = cos(th);

    float M = params[0];
    float a = params[1];
    float rho = params[2];
    float pphi = params[3];
    float B = params[4];
    float b = pphi;

    float x = 2*(fabs(l)-a)/(PI*M);

    int inside_condition = (fabs(l)>a ? 1 : 0); // 1 or 0 if condition is satisfied

    float r = rho + inside_condition * M * (x*atan(x) - 0.5*log(1+x*x));
    float drdl = atan(x) * 2/PI * (l>0 ? 1.0: -1.0) * inside_condition; 

    output[0] = 1; // time step iteration
    output[1] = pl; // dl/dt 
    output[2] = pth/(r*r); // dth/dt
    output[3] = b/(r*r*sth*sth); //  dphi/dt
    output[4] = B*B*drdl/(r*r*r);  // dpl/dt
    output[5] = b*b*cth/(r*r*sth*sth*sth); // dpth/dt

    // for(int i=0; i<6; ++i){
    //     if(isnan(output[i])){
    //         printf("Coordinate %d is nan, th: %f, r: %f, drdl: %f\n", i, th, r, drdl);
    //         printf("Params: M: %f, a: %f, rho: %f\n", M, a, rho);
    //         return false;
    //     }
    // }

    return true;
}

char __cuda_ode_func_name_d__[] = "Wormhole Differential Equation"; 



#endif


