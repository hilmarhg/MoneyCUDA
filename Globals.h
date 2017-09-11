#pragma once

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif 

const int ReadLimit = 7500;
const int ThreadsPerBlock = 512;// 128;// 
const int NofPaths = 65536;
const int NofT = 100;
const int NofOpts = 10;
const int NofParams = 5;
const int NofStrikes = 10;
const int BlocksPerOption = NofOpts*NofPaths / ThreadsPerBlock + 1;

typedef double Ty;