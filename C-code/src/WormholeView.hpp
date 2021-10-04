


#ifndef __wormhole_view_h__
#define __wormhole_view_h__

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "math.h"
#define PI 3.1415926535897932384626433832795028841971693993751058209749445923078

#include "cuErrorChecking.cuh"

#include "Skybox.hpp"


class WormholeView {
    public:    
        // sky1, sky2 are the universe at both sides of the wormhole
        // define image view resolution 
        WormholeView(Skybox &sky1, Skybox &sky2,  float fov=PI/2,
            int view_width=640, int view_height=480);
        ~WormholeView();

        void set_view(int view_width, int view_height);
        void set_fov(float fov);
        void set_resolution_scale(float resolution_scale);
        // void set_deflection(float *photon_param, float *end_coords, int n_points);
        // photon_param: list of starting photon position and direction
        // end_coords:

        void compute_deflection(float l0, float M, float rho, float a, float max_l);



        // return true if sucessfull
        bool get_texture_array(uint8_t* texture_array);
        bool write_image_jpg(const char* filename);


    protected:
        Skybox _sky1;
        Skybox _sky2;
        int _view_width, _view_height;
        

        bool is_deflection_computed=false;
        float _fov;

        // _int: coordinates retrieved from RK integration
        // _photon_theta_start is assumed from linear split of interval [0,PI[ into n_points
        int _int_n_points=0;
        float *_int_sign_l_end;
        float *_int_photon_theta_end;
        // float* all_coords;
    private:
        int n_points; 
        float _resolution_scale=2;


};

#endif
