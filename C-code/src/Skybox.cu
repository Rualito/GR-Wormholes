
#include "Skybox.hpp"

#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>


// a fudge parameter because I am lazy
#ifndef SKYCUBE_FILETYPE
#define SKYCUBE_FILETYPE ".jpg" 
#endif

#define IMG_INDEX(i,j,k, width, height, channels) ((j)*(width)*(channels)+(i)*(channels)+(k))


Skybox::Skybox(const char* filename, bool is_cube, float offset_phi, float offset_th, bool cuda_speedup){
    __is_cube = is_cube;
    __cuda_speedup = cuda_speedup;
    __offset_phi = offset_phi;
    __offset_th = offset_th;

    if(is_cube){
        __img = NULL;

        // its assumed they all have the same dimensions
        // if this is not true, then its all fucked
        // __img_up = stbi_load((String(filename) + "/top" + SKYCUBE_FILETYPE), &__width, &__height, &__channels, 0);
        // __img_down = stbi_load((String(filename) + "/bottom" + SKYCUBE_FILETYPE), &__width, &__height, &__channels, 0);
        // __img_front = stbi_load((String(filename) + "/front" + SKYCUBE_FILETYPE), &__width, &__height, &__channels, 0);
        // __img_back = stbi_load((String(filename) + "/back" + SKYCUBE_FILETYPE), &__width, &__height, &__channels, 0);
        // __img_left = stbi_load((String(filename) + "/left" + SKYCUBE_FILETYPE), &__width, &__height, &__channels, 0);
        // __img_right = stbi_load((String(filename) + "/right" + SKYCUBE_FILETYPE), &__width, &__height, &__channels, 0);

        this->initialize_image((String(filename) + "/top" + SKYCUBE_FILETYPE), __img_up, __d_img_up);
        this->initialize_image((String(filename) + "/bottom" + SKYCUBE_FILETYPE), __img_down, __d_img_down);
        this->initialize_image((String(filename) + "/front" + SKYCUBE_FILETYPE), __img_front, __d_img_front);
        this->initialize_image((String(filename) + "/back" + SKYCUBE_FILETYPE), __img_back, __d_img_back);
        this->initialize_image((String(filename) + "/left" + SKYCUBE_FILETYPE), __img_left, __d_img_left);
        this->initialize_image((String(filename) + "/right" + SKYCUBE_FILETYPE), __img_right, __d_img_right);

        // if(__img_up == NULL 
        //     || __img_down == NULL 
        //     || __img_front == NULL 
        //     || __img_back == NULL
        //     || __img_left == NULL 
        //     || __img_right == NULL){
        //     printf("Error loading some skycube image\n");
        //     exit(1);
        // }

        // if(__cuda_speedup){
        //     cudaMalloc((void**) &__d_img_up, __width*__height*__channels*sizeof(uint8_t));
        //     cudaMemcpy(__d_img_up, __img_up, __width*__height*__channels*sizeof(uint8_t), cudaMemcpyHostToDevice);


        // }


    } else {
        __img_up = NULL;
        __img_down = NULL;
        __img_front = NULL;
        __img_back = NULL;
        __img_left = NULL;
        __img_right = NULL;

        // __img = stbi_load(filename, &__width, &__height, &__channels, 0);

        this->initialize_image(filename, __img, __d_img);

        // if(__img == NULL){
        //     printf("Error loading image\n");
        //     exit(1);
        // }

        // if(__cuda_speedup){
        //     cudaMalloc((void**) &__d_img, __width*__height*__channels*sizeof(uint8_t));
        //     cudaMemcpy(__d_img, __img, __width*__height*__channels*sizeof(uint8_t), cudaMemcpyHostToDevice);
        // }

    }

}

void Skybox::initialize_image(const char* filename, uint8_t* img, uint8_t* d_img){
    img = stbi_load(filename, &__width, &__height, &__channels, 0);

    if(img == NULL){
        printf("Error loading image: %s\n", filename);
        exit(1);
    }

    if(__cuda_speedup){
        cudaMalloc((void**) &d_img, __width*__height*__channels*sizeof(uint8_t));
        cudaMemcpy(d_img, img, __width*__height*__channels*sizeof(uint8_t), cudaMemcpyHostToDevice);
    }
}

Skybox::~Skybox(){

    // Program may die in this distructor, remove free if that happens
    if(__img) free(__img);
    if(__img_up) free(__img_up);
    if(__img_down) free(__img_down);
    if(__img_front) free(__img_front);
    if(__img_left) free(__img_left);
    if(__img_right) free(__img_right);

    if(__d_img) cudaFree(__d_img);
    if(__d_img_up) cudaFree(__d_img_up);
    if(__d_img_down) cudaFree(__d_img_down);
    if(__d_img_front) cudaFree(__d_img_front);
    if(__d_img_back) cudaFree(__d_img_back);
    if(__d_img_left) cudaFree(__d_img_left);
    if(__d_img_right) cudaFree(__d_img_right);
}


// phi, th rotstion values for each face
// operation: rotate face back to front
__constant__ float rot_F[2]{0,0};
__constant__ float rot_B[2]{-PI, 0};

__constant__ float rot_R[2]{PI/2, 0};
__constant__ float rot_L[2]{-PI/2, 0};

__constant__ float rot_U[2]{0, PI/2};
__constant__ float rot_D[2]{0, -PI/2};

// __constant__ float 


// phi is along the width, x; th is along height, y
void Skybox::get_pixel(float phi, float th, uint8_t* pixel){
    
    float angle_phi = interval_mod(phi + __offset_phi, -PI, PI);
    float angle_th = interval_mod(th + __offset_th, 0, PI);

    if(__is_cube){
        // TODO: needs some algebra and rays inside a cube
        // referential at center of cube
        // x points FRONT
        // y points LEFT
        // z points UP


        // x = r cos phi sin th
        // y = r sin phi sin th
        // z = r cos th

        // faces: F, B, R, L, U, D
        // front, back, left, right, up, down
        // F: x = 1, y,z<1
        // B: x=-1, y,z<1
        // R (L): y=(-)1, x,z<1 
        // U (D): z=(-)1, z,y<1

        // xr = x/r, yr = y/r, zr = z/r
        float xr = cos(angle_phi) * sin(angle_th);
        float yr = sin(angle_phi) * sin(angle_th);
        float zr = cos(angle_th);

        // figure out if ray pointing to up face
        // assume z = +-1 -> r = +-1/(zr)
        // x = xr*r = +- xr/zr
        // y = yr * r = +- yr/zr 
        
        // if |xr/zr| < 1 and |yr/zr| < 1 
        // then we are in up or down face
        // up face : 0<th<PI/2
        // down face: PI/2<th<0

        // otherwise
        float transformed_phi = angle_phi;
        float transformed_th = angle_th;
        bool back_face = false;
        uint8_t* __img__;
        if(fabs(xr/zr)<1 & fabs(yr/zr)<1){
            // we are in UP or DOWN face

            if(angle_th<PI/2){ // UP
                transformed_phi = angle_phi + rot_U[0];
                transformed_th = angle_th + rot_U[1];
                __img__ = __img_up;
            } else { // DOWN
                transformed_phi = angle_phi + rot_D[0];
                transformed_th = angle_th + rot_D[1];
                __img__ = __img_down;
            }
        } else if(fabs(angle_phi) < PI/4) { // FRONT
            transformed_phi = angle_phi + rot_F[0];
            transformed_th = angle_th + rot_F[1];
            __img__ = __img_front;
        } else if(fabs(angle_phi-PI/2) < PI/4){ // RIGHT
            transformed_phi = angle_phi + rot_R[0];
            transformed_th = angle_th + rot_R[1];
            __img__ = __img_right;
        } else if(fabs(angle_phi+PI/2) < PI/4) { // LEFT
            transformed_phi = angle_phi + rot_R[0];
            transformed_th = angle_th + rot_R[1];
            __img__ = __img_left
        } else { // BACK
            transformed_phi = angle_phi + rot_B[0];
            transformed_th = angle_th + rot_B[1];
            back_face = true;
            __img__ = __img_back;
        }
        // y = r sin phi sin th
        // z = r cos th
        // cube within -1,1 in all axis
        // face projection onto y-z plane
        float tr_y = sin(transformed_phi) * sin(transformed_th);
        float tr_z = cos(transformed_th);

        // (tr_y + 1)/2 is between 0 and 1
        // multiplied by __width gives the pixel position

        // (1-2*back_face) flips the sign when back_face is true
        // this, with -__width*back_face in both indexes is intended to rotate the image 180 degrees
        int i = (1-2*back_face)*(-__width * back_face + (int) round( (tr_y+1)/2)*__width);
        int j = (1-2*back_face)*(-__height * back_face + (int) round( (tr_z+1)/2)*__height);
        // ASSUMPTION: all faces of the cube have the same __width, __height and __channels
        for(int k=0; k<__channels; ++k){
            pixel[k] = __img__[IMG_INDEX(i,j,k, __width, __height, __channels)];
        }

    } else {
        __img__ = __img;
        int i = (int)round( interval_mod(angle_phi, 0, 2*PI)* width/(2*PI)) ;
        int j = (int)round(interval_mod(angle_th, 0, PI) * height/PI);

        for(int k=0; k<__channels; k++){
            pixel[k] = __img[IMG_INDEX(i, j, k, __width, __height, __channels)];
        }
    }


}


