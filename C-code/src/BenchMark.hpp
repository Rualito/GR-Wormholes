#ifndef H_BENCHMARK_H
#define H_BENCHMARK_H

#include <ctime>

double BenchMark(char test = 0, char channel  = 0);

#ifdef BENCHMARK

#include <cstdio>
#define BENCHMARK_START(ch) BenchMark(0,ch)
#define BENCHMARK_END(ch) ( ch>=0 ? printf("[%s] ch:%d dt:%f\n", __PRETTY_FUNCTION__, ch, BenchMark(1, ch)): printf("[%s] ch:%d dt:%f", __PRETTY_FUNCTION__, ch, BenchMark(1, -ch)))

#else 

#define BENCHMARK_START
#define BENCHMARK_END

#endif



#endif
