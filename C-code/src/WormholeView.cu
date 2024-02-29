#include "WormholeView.cuh"

// #define BENCHMARK
#include "BenchMark.hpp"


#define THREADS_PER_BLOCK 16 // 1 dimensional thread blocks
#define THREADS(N) dim3((THREADS_PER_BLOCK<N?THREADS_PER_BLOCK:N),1,1);
#define BLOCKS(N) dim3((N>THREADS_PER_BLOCK?N/THREADS_PER_BLOCK:1),1,1);

#define MAX_ITER 2048 // how many iterations until the simulation stops

// channels should always be 4
#define TEXTURE_INDEX(i,j,k, width, height, channels) ((j)*(width)*(channels)+(i)*(channels)+(k))

#define SIGN(x) ( (x) > 0) - ( (x) < 0) 

__host__ __device__ 
float interval_mod(float x, float x0, float x1){
    return fmod(x-x0, x1) + x0;
}


WormholeView::WormholeView(Skybox &sky1, Skybox &sky2,float fov ,int view_width, int view_height, float resolution_scale): _sky1(sky1), _sky2(sky2),_fov(fov), _view_width(view_width), _view_height(view_height), _resolution_scale(resolution_scale){
    // n_points = PI/fov * view_width * _resolution_scale;
    is_deflection_computed = false;
}


WormholeView::~WormholeView(){
    if(_int_sign_l_end) free(_int_sign_l_end);
    if(_int_photon_theta_end) free(_int_photon_theta_end);

}

void WormholeView::set_view(int view_width, int view_height){
    _view_width = view_width;
    _view_height = view_height;
}

void WormholeView::set_fov(float fov){
    _fov = fov;
    
}

void WormholeView::set_resolution_scale(float resolution_scale){
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
    _int_n_points = (int)(PI/_fov * _view_width * _resolution_scale);
    int n_points = _int_n_points;

    if(_int_n_points != prev_n){
        if(_int_sign_l_end) free(_int_sign_l_end);
        if(_int_photon_theta_end) free(_int_photon_theta_end);
    }
    // allocate memory for the simulation data arrays
    _int_sign_l_end = (int8_t*) malloc(_int_n_points*sizeof(int8_t));
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
        float pth0 = p*sin(th_p)*cos(phi_p); // = observer p_x

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

    // Theta_in : (PI*i)/n_points
    // Theta_out : integration_results[n_coords*MAX_ITER*i + j*n_coords + k] ; theta: k=2 ; j=integration_end[i]-1
    // s_l : sign(" in results "); l : k=1; j = integration_end[i] -1

    // PARSE RESULTS INTO _int_sign_l_end AND _int_photon_theta_end
    
    for(int i=0; i<n_points; ++i){
        int k_th = 2;
        int k_l = 1;
        int j = integration_end[i]-1;
        _int_sign_l_end[i] = SIGN(integration_results[n_coords*MAX_ITER*i + j*n_coords + k_l]);
        _int_photon_theta_end[i] = integration_results[n_coords*MAX_ITER*i + j*n_coords + k_th];
    }

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


bool WormholeView::get_texture_array(uint8_t* texture, float look_th=PI/2, float look_phi=0) {
    // texture_array has alocated view_width*view_height*channels
    // Default look: wormhole right in front - z axis is always vertical 
    // Need to first convert look angles into deflection angles
    // deflection angles denominated by _tilde: th_tilde, phi_tilde.
    if(!is_deflection_computed){ 
        printf("Deflection angles not computed yet!\n");
        return false;
    }

    float Dphi = _fov;
    float L = _view_width/(2*tan(Dphi/2)); 

    // do this for each pixel on the screen
    for(int x = 0; x<_view_width; ++x){ // phi in look frame
        for(int y = 0; y<_view_height; ++y){  // th in look frame
            float xp = x-_view_width/2.0f;
            float yp = y-_view_height/2.0f;

            float dphi = atan2(xp,L);
            float dth = atan2(yp,L);

            float ray_th = look_th + dth;
            float ray_phi = look_phi + dphi;

            // converting angles to 'deflection' frame of reference
            float th_tilde = acos( cos(ray_phi)*sin(ray_th));
            float phi_tilde = interval_mod(atan2(cos(ray_th), sin(ray_phi)*sin(ray_th)), 0, 2*PI);
            
            // spherical symmetry
            float phi_tilde_prime = phi_tilde;

            int i0 = floor(th_tilde * n_points/PI);
            int i1 = ceil(th_tilde * n_points/PI);

            float d_th = PI/n_points;
            
            float pi0 = (th_tilde - (i0*PI/n_points)) / d_th;
            float pi1 = 1-pi0;

            // such that th_tilde = pi0*(i0*PI/n_points) + pi1*(i1*PI/n_points);

            // linear interpolation of previously computed deflection angles
            // in _tilde coordinates
            float th_tilde_prime = pi0*_int_photon_theta_end[i0] + pi1 * _int_photon_theta_end[i1];
            float sl = pi0*_int_sign_l_end[i0] + pi1 * _int_sign_l_end[i1];


            // Convert the angles back to the 'look' reference frame
            float th_prime = acos(sin(th_tilde_prime)*sin(phi_tilde_prime));
            float phi_prime = interval_mod(atan2(sin(th_tilde_prime)*cos(phi_tilde_prime),cos(th_tilde_prime)), 0, 2*PI);

            // pixel of the current look angle
            // uint8_t pixel[_sky1.channels];
            uint8_t pixel1[_sky1.__channels];
            uint8_t pixel2[_sky2.__channels];
            
            _sky1.get_pixel(phi_prime,th_prime,pixel1);
            _sky2.get_pixel(phi_prime,th_prime,pixel2);

            for(int k=0; k<_sky1.__channels;++k){
                // weighted pixel between horizon, if its on a switch zone
                // pixel[k] = pixel1[k] * (sl+1)*0.5 + (1-(sl+1)*0.5) * pixel2[k];
                texture[TEXTURE_INDEX(x, y, k, _view_width, _view_height, _sky1.__channels)] = pixel1[k] * (sl+1)*0.5 + (1-(sl+1)*0.5) * pixel2[k]; 
            }
        }
    }

    return true;
}


bool WormholeView::get_texture_array_CUDA(uint8_t* texture, float look_th, float look_phi) {
    if(!is_deflection_computed){ 
        printf("Deflection angles not computed yet!\n");
        return false;
    }

    // do _view_width threads loading the pixels for the image 


    return true;
}