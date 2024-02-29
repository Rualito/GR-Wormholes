#ifndef __sky_box_h__
#define __sky_box_h__

#include "math.h"
#define PI 3.1415926535897932384626433832795028841971693993751058209749445923078

#include <stdio.h>
#include <stdint.h>
#include "cuErrorChecking.cuh"

// #include <sys/types.h>
// #include <sys/stat.h>
// #include <unistd.h>

// int isDir(const char* fileName)
// {
//     struct stat path;

//     stat(fileName, &path);

//     return S_ISREG(path.st_mode);
// }


class Skybox {

    public:
        // filename can be either directory (for cube skybox) or an image file
        // loads the image file into a pointer
        // if cuda_speedup is true, the image is loaded into the GPU to paralelise pixel retrieval
        Skybox(const char* filename, bool is_cube=false, float offset_phi=0, float offset_th=0, bool cuda_speedup=false);
        Skybox(const Skybox &skybox);

        ~Skybox();
        
        void get_pixel(float phi, float th, uint8_t* pixel);
        // phi and th are arrays of n_points,  
        void get_pixel_CUDA(float* phi, float* th, int n_points, uint8_t* pixel);

        // TODO : 
        void set_offset_angles(float offset_phi=0, float offset_th=0);
    
    // protected:
        int __width, __height, __channels;

    private: 
        void initialize_image(const char* filename, uint8_t* img, uint8_t* d_img);
        void copy_image(uint8_t *_img_, uint8_t* _img_origin, uint8_t* _d_img_);
        float __offset_phi;
        float __offset_th;

        bool __is_cube;
        bool __cuda_speedup;
        
        uint8_t* __img;
        uint8_t* __d_img;
        
        uint8_t* __img_up;
        uint8_t* __img_down;
        uint8_t* __img_front;
        uint8_t* __img_back;
        uint8_t* __img_left;
        uint8_t* __img_right;

        uint8_t* __d_img_up;
        uint8_t* __d_img_down;
        uint8_t* __d_img_front;
        uint8_t* __d_img_back;
        uint8_t* __d_img_left;
        uint8_t* __d_img_right;
};

// places x inside interval [x0, x1[  


__global__ void extract_pixel_cuda(uint8_t* __img, // single image
    uint8_t* _img_u, // cube up
    uint8_t* _img_d, // cube down
    uint8_t* _img_f, // cube front
    uint8_t* _img_b, // cube back
    uint8_t* _img_l, // cube left
    uint8_t* _img_r, // cube right
    int __width, int __height, int __channels,
    float* phi, float* th, int n_points, uint8_t* pixel, float __offset_phi, float __offset_th, bool __is_cube);

#endif