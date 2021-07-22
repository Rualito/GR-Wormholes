#include <ctime>

#include "BenchMark.hpp"

#define MAX_CHANNEL 64

double BenchMark(char test, char channel)
{
    static clock_t start[MAX_CHANNEL] = { 0 };

    if(test == 0){
	start[channel] = clock();
	return -1;
    }

    return (clock()-start[channel])/((double) CLOCKS_PER_SEC);
}


    
    

