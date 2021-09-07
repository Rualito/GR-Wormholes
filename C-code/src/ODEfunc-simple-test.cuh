


// An ODE func ( f(n_param, *params, n_coords, *coords, *output) )
// that takes in physical parameters (like mass, radii, etc) 
// and coordinate points (could be position and/or momentum)
// in any coordinate system
// output has size n_coords
// returns bool : true if successful 

// typedef bool (*ODEfunc)(int n_param, float* params, int n_coords, float* coords, float* output);

#ifndef __cuda_ode_func__

// simple test for the cuda function
// harmonic oscillator
__device__
bool __cuda_ode_func__(int n_params, float *params, int n_coords, float* coords, float *output){
    if (n_params<1 | n_coords<2){
        return false;
    }

    output[0] = 1; // time step iteration
    output[1] = coords[2]; //dx/dt = y 
    output[2] = -params[0]*params[0] * coords[1]; //  dy/dt = -w^2 * x
    // these are equations for the harmonic oscillator

    return true;
}

char __cuda_ode_func_name_d__[] = "Harmonic Oscillator"; 



#endif


