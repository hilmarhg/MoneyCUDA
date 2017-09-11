#pragma once

#include "Globals.h"
#include "CudaVector.h"
#include "OptionData.h"


extern "C" Ty* _cudaMalloc(Ty* d_Arr, size_t Size);
extern "C" cudaError_t _cudaMemcpyHtoD(void* d_Arr, const void* h_Arr, size_t Size);
extern "C" cudaError_t _cudaMemcpyDtoH(Ty* h_Arr, Ty* d_Arr, size_t Size);
extern "C" cudaError_t _cudaFree(Ty* d_Arr);
extern "C" cudaError_t _cudaDeviceReset();
extern "C" void RandomKernelTestWrapper(Ty params, Ty* resarr);
extern "C" void HestonKernelWrapper(const Ty* d_params, const OptionData* h_optiondatalist, const OptionData* d_optiondatalist, Ty* h_array, Ty* d_array, CudaVector* resarr, const int size);

