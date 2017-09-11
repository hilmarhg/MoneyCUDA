#pragma once
#include <iostream>
#include <math.h>
#include <vector>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include "SimulationKernels.h"
#include "BiasGauge.h"
#include "CmdlDebug.h"

using namespace std;

__global__
void RandomKernelTest(Ty* results, Ty* exresults)
{
	__shared__ Ty blockmemory[ThreadsPerBlock];
	unsigned int totali = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int i = threadIdx.x;
	Ty S0 = 100;
	Ty R = 0.0;
	Ty mat = 0.5;
	Ty strike = 90;
	Ty otype = 1.0;
	Ty dt = mat / (Ty)NofT;
	Ty kappa = 2.0;
	Ty theta = 0.02;
	Ty V0 = 0.015;
	Ty sigma = 0.6;
	Ty rho = -0.7;

	if (totali < NofPaths*NofOpts) 
	{
		curandState_t localstate;
		curand_init(totali * 2, 0, 0, &localstate);
		Ty St = S0;
		Ty Vt = V0;
		Ty e1out = 0;
		Ty e2out = 0;
		for (int t = 0; t < NofT; t++)
		{
			float2 x = curand_normal2(&localstate);
			Ty e1 = x.x;
			Ty e2 = x.y;
			Ty e3 = rho*e1 + sqrt(1 - rho*rho)*e2;
			Vt = Vt + kappa*(theta - Vt)*dt + sigma*sqrt(Vt*dt)*e1;
			Vt = (Vt > 0) ? Vt : 0;
			St = St*exp((R - 0.5*Vt)*dt + sqrt(Vt*dt)*e2);
			e1out = e1;
			e2out = e3;
		}
		results[totali] = e1out;
		exresults[totali] = e2out;
	}
	__syncthreads();
}

void RandomKernelTestWrapper(const Ty params, Ty* resarr)
{
	CmdlDebug cld;
	int size = sizeof(Ty)*NofPaths * NofOpts;
	int ex_size = sizeof(Ty) * NofPaths * NofOpts;
	cudaError_t err = cudaSuccess;
	Ty* h_array = (Ty *)malloc(size);
	Ty* h_exarray = (Ty *)malloc(ex_size);
	Ty* d_array = NULL;
	Ty* d_exarray = NULL;
	cld.err.push_back(make_pair("d_array Malloc", cudaMalloc((void**)&d_array, size)));
	cld.err.push_back(make_pair("d_exarray Malloc", cudaMalloc((void**)&d_exarray, ex_size)));
	RandomKernelTest<<<BlocksPerOption * NofOpts, ThreadsPerBlock>>>(d_array, d_exarray);
	cld.err.push_back(make_pair("Host to Device copy",cudaMemcpy(h_array, d_array, size, cudaMemcpyDeviceToHost)));
	cld.err.push_back(make_pair("Host to Device ex copy",cudaMemcpy(h_exarray, d_exarray, ex_size, cudaMemcpyDeviceToHost)));
	std::vector<Ty> optvals(NofOpts, 0);
	cld.err.push_back(make_pair("Host to Device copy",cudaMemcpy(h_array, d_array, size, cudaMemcpyDeviceToHost)));
	cld.err.push_back(make_pair("Host to Device ex copy",cudaMemcpy(h_exarray, d_exarray, ex_size, cudaMemcpyDeviceToHost)));
	cld.ErrorCheck();
	Ty covar = BiasGauge::Correlation(h_array, h_exarray, NofPaths * NofOpts);
	std::cout << "covar " << covar << std::endl;
}

__global__
void HestonKernel(const Ty* params, const OptionData* optdatalist, Ty* paths, Ty* exresults)
{
	unsigned int totali = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int i = threadIdx.x;
	int optindex = totali / (NofPaths);
	Ty S0 = optdatalist[optindex].ap;
	Ty R = optdatalist[optindex].rf;
	Ty Q = 0.0;
	Ty mat = optdatalist[optindex].mat;
	Ty dt = mat / (float)NofT;
	Ty kappa = params[0];
	Ty theta = params[1];
	Ty V0 = params[2];
	Ty sigma = params[3];
	Ty rho = params[4];
	Ty St = S0;
	Ty Vt = V0;

	if (totali < NofPaths*NofOpts) 
	{
		curandState_t localstate;
		curand_init(totali * 100, 0, 0, &localstate);
		Ty timet = 0;
		for (int t = 0; t < NofT; t++)
		{
			float2 x = curand_normal2(&localstate);
			Ty e1 = x.x;
			Ty e2 = x.y;
			Ty e3 = rho*e1 + sqrt(1.0 - rho*rho)*e2;
			St = St*exp((R - Q - 0.5*Vt)*dt + sqrt(Vt*dt)*e3);
			Vt = Vt + kappa*(theta - Vt)*dt + sigma*sqrt(Vt*dt)*e1;
			Vt = (Vt > 0) ? Vt : 0.00001;
		}
	}

	if (totali < 2)
	{
		exresults[1] = totali;
		exresults[2] = V0;
		exresults[3] = S0;
	}
	paths[totali] = St;
}

void HestonKernelWrapper(const Ty* d_params, const OptionData* h_optiondatalist, const OptionData* d_optiondatalist, Ty* h_array, Ty* d_array, CudaVector* resarr, const int size)
{
	CmdlDebug cld;
	cudaError_t err = cudaSuccess;
	Ty* d_exarray = NULL;
	Ty* h_exarray = (Ty *)malloc(sizeof(Ty) * 4);
	cld.err.push_back(make_pair("d_exarray Malloc",cudaMalloc((void**)&d_exarray, sizeof(Ty) * 4)));
	HestonKernel<<<BlocksPerOption, ThreadsPerBlock>>>(d_params, d_optiondatalist, d_array, d_exarray);
	cld.err.push_back(make_pair("Host to Device copy",cudaMemcpy(h_array, d_array, size, cudaMemcpyDeviceToHost)));
	cld.err.push_back(make_pair("Host to Device ex copy",cudaMemcpy(h_exarray, d_exarray, sizeof(Ty) * 4, cudaMemcpyDeviceToHost)));
	cld.ErrorCheck();
	std::vector<CudaVector> optvalslist(NofOpts);
	Ty otype, strike, value, st;

	for (int t = 0; t < NofOpts; t++)
	{
		for (int i = 0; i < NofPaths; i++)
		{
			st = h_array[t*NofPaths + i];
			for (int k = 0; k < NofStrikes; k++)
			{
				otype = h_optiondatalist[t].OT[k];
				strike = h_optiondatalist[t].SL[k];
				if (otype > 0)
				{
					value = (st - strike > 0) ? st - strike : 0;
					//callcount += 1;
				}
				else if (otype < 0)
				{
					value = (strike - st > 0) ? strike - st : 0;
					//putcount += 1;
				}
				optvalslist[t].data[k] += value;
			}
		}
	}

	for (int opt = 0; opt < NofOpts; opt++)
		resarr[opt] = optvalslist[opt] / (Ty)(NofPaths);
	
}

