#include <stdio.h>
#include <stdlib.h>

#include <SFML/Graphics.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "math.h"
#define PI 3.1415926535897932384626433832795028841971693993751058209749445923078

// THIS WORKS, DON'T TWEAK THIS
#define IMG_INDEX(i,j,k, width, height, channels) ((j)*(width)*(channels)+(i)*(channels)+(k))
// channels should always be 4
#define TEXTURE_INDEX(i,j,k, width, height, channels) ((j)*(width)*(channels)+(i)*(channels)+(k))

// #define BENCHMARK
#include "BenchMark.hpp"

// #define DEBUG
// #define COUNT
#include "Debug.hpp"

int main(int argc, char** argv){

    


    return 0;
}