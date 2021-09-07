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

double where_on_interval(double x, double x_min,double x_max){
    return abs(fmod(((x-x_min) / (x_max-x_min)), 1.0));
}

// This needs some tweaks around fmod (sign is different from python)
double move_to_interval(double x, double x_min, double x_max){
    return fmod((x-x_min), (x_max-x_min) + x_min);
}

uint8_t* get_pixel(uint8_t *angle_img, int width, int height, int channels, double phi, double th){
    double x = where_on_interval(phi, -PI, PI);
    double y = where_on_interval(th, 0, PI);

    int i = (int)(x*width);
    int j = (int)(y*height);
    // Gets only the pointer to the first pixel component value
    int idx = IMG_INDEX(i, j, 0,  width, height, channels);
    //printf("phi: %f, th: %f, x: %f, y: %f\n", phi, th, x, y);
    //printf("i: %d, j:%d, idx: %d, width: %f, height: %f\n", i,j,idx);
    
    if(idx>=width*height*channels){
        printf("Reporting this to the authorities: %d %d %d\n", i,j,0);
        exit(1);
    }
    return &(angle_img[idx]);
}

// This function should be paralelized with cuda
void generate_view(uint8_t *angle_img, int width, int height, int channels, 
    double phi0, double th0, uint8_t *view,  int view_width, int view_height, double fov, uint8_t* texture_pixels=NULL, bool onlyTexture=false){
    ECHOS;
    double Dphi = fov;
    // double Dth = (height*fov)/width;
    
    double distance = view_width/(2*tan(Dphi/2));
    // uint8_t *pixel;
    // pixel = (uint8_t*) malloc(channels*sizeof(uint8_t));
    int largest_idx = 0;
    if(!onlyTexture){
        for(int y=0; y<view_height; ++y){
            for(int x=0; x<view_width; ++x){
                // printf("On loop: %d %d \n", x, y);
                double xp = x-view_width/2.0;
                double yp = y-view_height/2.0;
                
                double dphi = atan(xp/distance);
                double dth = atan(yp/distance);
                
                double pphi = phi0+dphi;
                double pth = th0 + dth;
                
                uint8_t *pixel = get_pixel(angle_img, width, height, channels, pphi, pth);
                
                for(int z=0; z<channels; ++z){
                    int idx = IMG_INDEX(x,y,z,view_width,view_height,channels);
                    // int idp = IMG_INDEX(x%width,y%height,z, width, height,channels);
                    if(idx>largest_idx){
                        largest_idx = idx;
                    }
                    if(idx>=view_width*view_height*channels){
                        printf("Reporting this to the authorities: %d %d %d\n", x,y,z);
                        exit(1);
                    }
                    view[idx] = pixel[z];
                    //view[idx] = angle_img[idp];
                    
                }

            }
        }
    }
    if(texture_pixels){
        const int texture_channels = 4;
        if(channels<3){
            printf("Input image doesn't have enough color channels (RGB format required)\n");
            exit(-1);
        }
        for(int y=0; y<view_height; ++y){
            for(int x=0; x<view_width; ++x){
                // printf("On loop: %d %d \n", x, y);
                double xp = x-view_width/2.0;
                double yp = y-view_height/2.0;
                
                double dphi = atan(xp/distance);
                double dth = atan(yp/distance);
                
                double pphi = phi0+dphi;
                double pth = th0 + dth;
                
                uint8_t *pixel = get_pixel(angle_img, width, height, channels, pphi, pth);
                
                for(int z=0; z<channels; ++z){
                    int idx = TEXTURE_INDEX(x,y,z,view_width,view_height,texture_channels);
                    // int idp = TEXTURE_INDEX(x%width,y%height,z, width, height,channels);
                    if(idx>largest_idx){
                        largest_idx = idx;
                    }
                    if(idx>=view_width*view_height*texture_channels){
                        printf("Reporting this to the authorities: %d %d %d\n", x,y,z);
                        exit(-1);
                    }
                    texture_pixels[idx] = pixel[z];
                    //view[idx] = angle_img[idp];
                }
                // add a value for the opacity (textures only accept RGBA)
                int idx = TEXTURE_INDEX(x,y,(texture_channels-1),view_width,view_height,texture_channels);
                // int idp = TEXTURE_INDEX(x%width,y%height,z, width, height,channels);
                texture_pixels[idx] = 255;
            }
        }
    }

    ECHOF;
    // free(pixel);
}

void sfml_trianglesFromImage(uint8_t* img, int imgSize[3], int windowSize[2], sf::VertexArray& vex){
    ECHOS;
    // This assumes img_channels is 3 to define colors 
    // Window is windowSize[0] x windowSize[1]
    // imgSize is {width, height, channels}
    int img_width = imgSize[0];
    int img_height = imgSize[1];
    int img_channels = imgSize[2]; 
    double stepWidth = (windowSize[0]*1.)/(img_width);
    double stepHeight = (windowSize[1]*1.)/(img_height);
    
    // vex.clear();
    for(int i=0; i<img_width; ++i){
        for(int j=0; j<img_height; ++j){
            // printf("On loop: %d %d\n", i, j);
            // vertex array of triangles (that complete as quads)
            double px0 = i*stepWidth;
            double px1 = (i+1)*stepWidth;
        
            double py0 = j*stepHeight;
            double py1 = (j+1)*stepHeight;

            vex[j*img_width*6+i*6+0].position = sf::Vector2f(px0,py0); // pixel0
            vex[j*img_width*6+i*6+1].position = sf::Vector2f(px1,py0);// pixel1
            vex[j*img_width*6+i*6+2].position = sf::Vector2f(px0,py1);// pixel2

            vex[j*img_width*6+i*6+3].position = sf::Vector2f(px1,py0);// pixel1
            vex[j*img_width*6+i*6+4].position = sf::Vector2f(px0,py1);// pixel2
            vex[j*img_width*6+i*6+5].position = sf::Vector2f(px1,py1);// pixel3
            

            // vex[j*img_width*3+i*3+0].position = sf::Vector2f(px0,py0); // pixel0
            // vex[j*img_width*3+i*3+1].position = sf::Vector2f(px1,py0);// pixel1
            // vex[j*img_width*3+i*3+2].position = sf::Vector2f(px0,py1);// pixel2

            
            uint8_t *pixel0 = &(img[IMG_INDEX(i,j,0, img_width, img_height, img_channels)]);
            uint8_t *pixel1 = &(img[IMG_INDEX(i+1,j,0, img_width, img_height, img_channels)]);
            uint8_t *pixel2 = &(img[IMG_INDEX(i,j+1,0, img_width, img_height, img_channels)]);
            uint8_t *pixel3 = &(img[IMG_INDEX(i+1,j+1,0, img_width, img_height, img_channels)]);
             
            sf::Color colorp0(pixel0[0], pixel0[1], pixel0[2]); 
            sf::Color colorp1(pixel1[0], pixel1[1], pixel1[2]); 
            sf::Color colorp2(pixel2[0], pixel2[1], pixel2[2]); 
            sf::Color colorp3(pixel3[0], pixel3[1], pixel3[2]); 
            
            vex[j*img_width*6+i*6+0].color = colorp0; // pixel0
            vex[j*img_width*6+i*6+1].color = colorp1;// pixel1
            vex[j*img_width*6+i*6+2].color = colorp2;// pixel2

            vex[j*img_width*6+i*6+3].color = colorp1;// pixel1
            vex[j*img_width*6+i*6+4].color = colorp2;// pixel2
            vex[j*img_width*6+i*6+5].color = colorp3;// pixel3
            
            
            // vex[j*img_width*3+i*3+0].color = colorp0; // pixel0
            // vex[j*img_width*3+i*3+1].color = colorp1;// pixel1
            // vex[j*img_width*3+i*3+2].color = colorp2;// pixel2            
        }   

    }
    ECHOF;
}

void sfml_quadsFromImage(uint8_t* img, int imgSize[3], int windowSize[2], sf::VertexArray& vex){
    ECHOS;
    // This assumes img_channels is 3 to define colors 
    // Window is windowSize[0] x windowSize[1]
    // imgSize is {width, height, channels}
    int img_width = imgSize[0];
    int img_height = imgSize[1];
    int img_channels = imgSize[2]; 
    double stepWidth = (windowSize[0]*1.)/(img_width);
    double stepHeight = (windowSize[1]*1.)/(img_height);
    printf("StepW: %f, StepH: %f\n", stepWidth, stepHeight);

    // vex.clear();
    for(int j=0; j<img_height; ++j){
        for(int i=0; i<img_width; ++i){    
            // vertex array of quads 
            int px0 = (int)(i*stepWidth);
            int px1 = (int)((i+1)*stepWidth);
        
            int py0 = (int)(j*stepHeight);
            int py1 = (int)((j+1)*stepHeight);

            vex[j*img_width*4+i*4+0].position = sf::Vector2f(px0,py0); // pixel0
            vex[j*img_width*4+i*4+1].position = sf::Vector2f(px1,py0);// pixel1
            vex[j*img_width*4+i*4+2].position = sf::Vector2f(px1,py1);// pixel2
            vex[j*img_width*4+i*4+3].position = sf::Vector2f(px0,py1);// pixel3

            uint8_t *pixel0 = &(img[IMG_INDEX(i,j,0, img_width, img_height, img_channels)]);
            uint8_t *pixel1 = &(img[IMG_INDEX(i+1,j,0, img_width, img_height, img_channels)]);
            uint8_t *pixel2 = &(img[IMG_INDEX(i+1,j+1,0, img_width, img_height, img_channels)]);
            uint8_t *pixel3 = &(img[IMG_INDEX(i,j+1,0, img_width, img_height, img_channels)]);
             
            sf::Color colorp0(pixel0[0], pixel0[1], pixel0[2]); 
            sf::Color colorp1(pixel1[0], pixel1[1], pixel1[2]); 
            sf::Color colorp2(pixel2[0], pixel2[1], pixel2[2]); 
            sf::Color colorp3(pixel3[0], pixel3[1], pixel3[2]); 

            vex[j*img_width*4+i*4+0].color = colorp0; // pixel0
            vex[j*img_width*4+i*4+1].color = colorp1;// pixel1
            vex[j*img_width*4+i*4+2].color = colorp2;// pixel2
            vex[j*img_width*4+i*4+3].color = colorp3;// pixel3
        }
    }
    ECHOF;
}


int main(int argc, char ** argv){
    ECHOS;
    // window.clear();


    // sf::Text text;
    // text.setFont(mFont);
    // text.setString("Greetings ma dudes\n");
    // text.setCharacterSize(14);
    // text.setFillColor(sf::Color::Black);

    // text.setString("x: "+to_string(mpos.x)+"; y: "+to_string(mpos.y));
	// text.setPosition(10.,10.);


    // window.draw(text);
    // while(window.isOpen()){
    //     sf::Event event;
    //     while(window.pollEvent(event)){
    //         if(event.type == sf::Event::Closed) window.close();
    //     }

	//     window.display();
    // }

    
    // Load basic skybox image
    int width, height, channels;
    uint8_t *angle_img = stbi_load("../wormhole_horizon.jpg", &width, &height, &channels, 0);
    if(angle_img == NULL){
        printf("Error in loading image\n");
        exit(1);
    }

    printf("Loaded saturn w %d, h %d, c %d\n", width, height, channels);
    printf("Angle_img total size: %d\n", width*height*channels);

    // Setting view array
    uint8_t *view, *texture_pixels;

    int view_width = 1920;
    int view_height = 1080; 

    if(argc>2){
        view_width = atoi(argv[1]);
        view_height = atoi(argv[2]);
    }

    // these ones can have any format taken by the stb lib, used for saving to file
    view = (uint8_t*) malloc(sizeof(uint8_t)*view_width*view_height*channels);
    // these pixels have a RGBA format
    texture_pixels = (uint8_t*) malloc(sizeof(uint8_t)*view_width*view_height*4);

    double phi0 = -0.5*PI;
    double th0 = PI*0.5;
    double fov = PI*0.56;
    
    // Generating the first view
    BENCHMARK_START(0);
    generate_view(angle_img, width, height, channels, phi0, th0, view, view_width, view_height, fov, texture_pixels);
    BENCHMARK_END(0);  
    // generate an initial view, as it is relatively cheap in initialization
    stbi_write_jpg("window view.jpg", view_width, view_height, channels, view, 100);
    // sf::Image img_view;

    // if(!img_view.loadFromFile("window view.jpg")){
    //     printf("Failure loading image\n");
    // }

    double phi1 = phi0;
    double th1 = th0;

    int windowSize[2]{1280,720};
    // int viewSize[3]{view_width, view_height, channels};
    
    sf::RenderWindow window(sf::VideoMode(windowSize[0],windowSize[1]), "Hello world");
    
    // sf::VertexArray vex(sf::Triangles,2*3*view_width*view_height);
    // sf::VertexArray vexq(sf::Quads,4*view_width*view_height);

    sf::Texture texture;
    if(!texture.create(view_width, view_height)){
        printf("Failure in creating texture\n");
        exit(-1);
    }

    sf::Sprite sprite;
    texture.update(texture_pixels);
    sprite.setTexture(texture);
    // sprite.setScale(sX.., sY...);

    
    // Debug circle
    // sf::CircleShape circ(10);
    // circ.setPosition(0,50);
    // double circ_velx = 1;
    // double circ_posx = 0;
    // double circ_posy = 50;

    printf("View total_size: %d\n", view_width*view_height*channels);
    // sf::VertexArray vex(sf::Triangles,2*3*view_width*view_height);
    // sf::VertexArray vex(sf::Triangles,3*view_width*view_height);
    // Window main loop
    while(window.isOpen()){
        sf::Event event;
        while(window.pollEvent(event)){
            if(event.type == sf::Event::Closed) window.close();
            switch(event.type){
                // Keyboard event managing
                case sf::Event::KeyPressed:
                    if(event.key.code == sf::Keyboard::Up){
                        th0 -= 0.025*PI;
                    } else if (event.key.code == sf::Keyboard::Down){
                        th0 += 0.025*PI;
                    }else if (event.key.code == sf::Keyboard::Left){
                        phi0-=0.05*PI;
                    }else if (event.key.code == sf::Keyboard::Right){
                        phi0+=0.05*PI;
                    }
                    // Readjusting the angles if they pass some threshold
                    if(phi0<-PI) phi0+=PI;
                    if(phi0>PI) phi0-=PI;
                    if(th0<0) th0=0;
                    if(th0>PI) th0=PI;

                    break;
                default:
                    break;
            } 
        }
        // in case the window gets resized
        sf::Vector2u wsize = window.getSize();
        windowSize[0] = (int)wsize.x;
        windowSize[1] = (int)wsize.y;
        sprite.setScale((windowSize[0]+0.)/view_width,(windowSize[1]+0.)/view_height );
        // Generate new views based on the view direction angles 
        // Only computes this if needs new angle
        // Updates the view pointer and the texture
        if(th1 != th0 || phi1!=phi0){
            BENCHMARK_START(0);
            // onlyTexture set to true
            generate_view(angle_img, width, height, channels, phi0, th0, view, view_width, view_height, fov, texture_pixels, true);
            BENCHMARK_END(0);        
            // means that the view array doesn't get updated, not worth loading to file
            // stbi_write_jpg("window view.jpg", view_width, view_height, channels, view, 100);
            // if(!img_view.loadFromFile("window view.jpg")){
            //     printf("Failure loading image\n");
            // }
            // texture.update(img_view);
            texture.update(texture_pixels);
        }
        phi1 = phi0;
        th1 = th0;
        
        // Changes the vertex array vex based on the generated view 
        // dont use primitive vertex arrays, prone to mess up
        // sfml_trianglesFromImage(view, viewSize, windowSize, vex);
        // sfml_quadsFromImage(view, viewSize, windowSize, vexq);
        // printf("Vertex count: %d\n", vex.getVertexCount());
        // printf("Reference outside: %p\n",&vex);

        // circ_posx += circ_velx; 
        // circ_posy += 1.5;
        // if(circ_posx>windowSize[0]){ // circle loops around
        //     circ_posx -= windowSize[0];
        // }
        // if(circ_posy>windowSize[1]){ // circle loops around
        //     circ_posy -= windowSize[1];
        // }
        // circ.setPosition(circ_posx, circ_posy);

        window.clear();
        window.draw(sprite);
        // window.draw(circ);
        window.display();
        
        // printf("New window\n");
    }


    stbi_image_free(angle_img); // same as free
    free(view);
    free(texture_pixels);
    ECHOF;
    return 0;
}