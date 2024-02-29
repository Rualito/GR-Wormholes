#ifndef _IMAGE_H_
#define _IMAGE_H_

#include <stdint.h>

#include <string.h>

#include <sys/types.h>

class Image {
    public:
        Image();
        Image(char* filename);
        ~Image();

        void loadf(uint8_t*);

        uint8_t* getData() const { return _imgData; };
    private:
        uint8_t* _imgData;
};

#endif

