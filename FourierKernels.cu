
#include <iostream>
#include <complex>
#include <cuComplex.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <random>
#include <math.h>


//const double pi = 3.14159265359;
const int NofE = 256;
const int UIL = 100; //upperintegrallimit
//const int NofP = 20;
//const int ThreadsPerBlock = 256;
////typedef std::complex<double> dcomp;
//typedef thrust::complex<double> dcomp;
//typedef thrust::device_vector<double> dvector;
//typedef thrust::host_vector<double> hvector;

__host__ __device__
inline double indexer(int i)
{
	double epsilon = 0.00001;
	double nofe = (double)NofE;
	double uil = (double)UIL;
	return ((double)i / nofe)*((double)i / nofe)*uil + epsilon;
}

