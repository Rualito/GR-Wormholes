



#include "WormholeView.hpp"

// #define BENCHMARK
#include "BenchMark.hpp"

#define THREADS_PER_BLOCK 16 // 1 dimensional thread blocks
#define THREADS(N) dim3((THREADS_PER_BLOCK<N?THREADS_PER_BLOCK:N),1,1);
#define BLOCKS(N) dim3((N>THREADS_PER_BLOCK?N/THREADS_PER_BLOCK:1),1,1);



WormholeView::WormholeView(Skybox &sky1, Skybox &sky2,float fov ,int view_width=640, int view_height=480, float _resolution_scale=2): _sky1(sky1), _sky2(sky2),_fov(fov), _view_width(view_width), _view_height(view_height), _resolution_scale(resolution_scale){
    // n_points = PI/fov * view_width * _resolution_scale;
    is_deflection_computed = false;
}


WormholeView::~WormholeView(){
    if(_int_sign_l) free(_int_sign_l);
    if(_int_photon_theta_end) free(_int_photon_theta_end);

}

void WormholeView::set_view(int view_width, int view_height){
    _view_width = view_width;
    _view_height = view_height;
}

void WormholeView::set_fov(float fov){
    _fov = fov;
    
}

void WormholeView::set_resolution_scale(float resolution_scale){}int view_width, int view_height){
    _resolution_scale = resolution_scale;
}



void WormholeView::compute_deflection(float l0, float M, float rho, float a, float max_l){
// Calls the RK ODE integrator and obtains the position of the raytraced photon
// Geometry:
// observer in th=PI/2, phi=0, L=l0
// observer z axis aligned with L direction
//          x axis aligned with wormhole th=0 axis
//          y pointing towards growing phi
// th_p is the angle between observer photon and observer z
// phi_p is associated angle with observer x

    int prev_n = _int_n_points;
    _int_n_points = (int)(PI/_fov * _view_width * resolution_scale);

    if(_int_n_points != prev_n){
        if(_int_sign_l_end) free(_int_sign_l_end);
        if(_int_photon_theta_end) free(_int_photon_theta_end);
    }

    _int_sign_l_end = (float*) malloc(_int_n_points*sizeof(float));
    _int_photon_theta_end = (float*) malloc(_int_n_points*sizeof(float));

    const int n_coords = 6; 
    const int n_params = 5;

    float *coord_array = new float[n_points*n_coords]; 
    float *params_array = new float[n_points*n_params];
    float t0 = 0;
    float phi0 = 0;
    float th0 = PI/2;

    // loading ODE initial coordinates and parameters into array
    for(int i=0; i<_int_n_points; ++i){
        float p = 1;
        float th_p = (PI * i)/n_points;
        float phi_p = 0;

        float pl0 = p*cos(th_p); // = observer p_z
        float pphi0 = p*sin(th_p)*sin(phi_p); // = observer p_y
        float pth0 = p*sin(th_p)*cos(phi_p) // = observer p_x

        float sth = sin(th0);
        float B = sqrt(pth0*pth0 + pphi0*pphi0/(sth*sth));

        float coords[n_coords]{t0,l0,th0, phi0, pl0, pth0};
        float params[n_params]{M, a, rho, pphi0, B};

        for(int j=0; j<n_coords; ++j){
            coord_array[i*n_coords+j] = coords[j]; 
        }
        for(int j=0; j<n_params; ++j){
            params_array[i*n_params+j] = params[j]; 
        }
    }

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
    // printf("Before CUDA\n");
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

    // PARSE RESULTS INTO _int_sign_l_end AND _int_photon_theta_end
    // USE SOMETHING FROM THIS :
    
    // for(int i=0; i<n_points; ++i){ // for each requested point
    //     outdata << "New point coordinate " << i << "; params:\t";
    //     for(int j=0; j<n_params; ++j){
    //         float param_now = params_array[i*n_params+j];
    //         outdata << param_now << "\t";
    //     }
    //     outdata << "\n";
    //     for(int k=0; k<n_coords; ++k){
    //         float coord_now = coord_array[i*n_coords+k];
    //         outdata << coord_now << "\t";
    //     } 
    //     outdata <<  "\n";
    //     //printf("integration end: %d\n", integration_end[i]);
    //     for(int j=0; j<integration_end[i]; ++j){ // throughout the simulation results
    //         for(int k=0; k<n_coords; ++k){
    //             outdata << integration_results[n_coords*MAX_ITER*i + j*n_coords+k] << "\t";
    //             // 3 coordinates
    //         }
    //         outdata << "\n";
    //     }
    //     outdata << "\n";
    // }

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

    is_deflection_computed = true;
}
