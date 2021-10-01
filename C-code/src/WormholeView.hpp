


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

#define THREADS_PER_BLOCK 16 // 1 dimensional thread blocks
#define THREADS(N) dim3((THREADS_PER_BLOCK<N?THREADS_PER_BLOCK:N),1,1);
#define BLOCKS(N) dim3((N>THREADS_PER_BLOCK?N/THREADS_PER_BLOCK:1),1,1);



class WormholeView {
    public:    
        // sky1, sky2 are the universe at both sides of the wormhole
        WormholeView(Skybox sky1, Skybox sky2, int view_width=640, int view_height=480);

        void set_resolution(int view_width, int view_height);
        void set_deflection(float *photon_param, float *end_coords);

        bool get_texture_array(uint8_t* texture_array);
        bool write_image_jpg(const char* filename);

        ~WormholeView();
    protected:
        int width1;
        int height1;
        int channels1;

        int width2;
        int height2;
        int channels2;

        bool is_deflection_set=false;

        float* all_coords;
    private:

};

#endif
