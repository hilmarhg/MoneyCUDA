#pragma once
#include <iostream>
#include <cuda_runtime.h>
#include <Eigen/Dense>
#include "cppoptlib/meta.h"
#include "cppoptlib/problem.h"
#include "cppoptlib/solver/bfgssolver.h"
#include "SimulationKernels.h"
#include "CmdlDebug.h"

using namespace std;
using namespace cppoptlib;

class ParallelizedObjective : public Problem <Ty>, CmdlDebug
{
public:
	void Initialize(const OptionData odlist[])
	{
		itercount = 0;
		Ty initparams[5] = { 0., 0., 0., 0. };
		params = initparams;
		h_ODlist = (OptionData*)malloc(sizeof(OptionData)*NofOpts);
		memcpy(h_ODlist, odlist, sizeof(OptionData)*NofOpts);
		size = sizeof(Ty)*NofPaths*NofOpts;
		ex_size = sizeof(Ty) * BlocksPerOption * NofOpts;
		err.push_back(make_pair("FirstInit", cudaSuccess));
		h_array = (Ty*)malloc(size);
		Ty* h_exarray = (Ty *)malloc(ex_size);
		d_array = NULL; //this is for the results, i.e. the devicetohost copy
		Ty* d_exarray = NULL;
		err.push_back(make_pair("DeviceArrayMalloc", cudaMalloc((void**)&d_array, size)));
		err.push_back(make_pair("DeviceExtraArrayMalloc", cudaMalloc((void**)&d_exarray, ex_size)));
		err.push_back(make_pair("DeviceParamsMalloc", cudaMalloc((void**)&d_params, sizeof(Ty) * 5)));
		err.push_back(make_pair("DeviceOptionDataMalloc", cudaMalloc((void**)&d_ODlist, sizeof(OptionData)*NofOpts))); //d_ODlist is for the input into the kernel wrapper, i.e. hosttodevice copy
		err.push_back(make_pair("CopyParamsToDevice", cudaMemcpy(d_params, params, sizeof(Ty) * 5, cudaMemcpyHostToDevice)));
		err.push_back(make_pair("CopyOptionDataToDevice", cudaMemcpy(d_ODlist, odlist, sizeof(OptionData)*NofOpts, cudaMemcpyHostToDevice)));
		Ty* h_paramlist = NULL;
		h_paramlist = (Ty*)malloc(sizeof(Ty) * 5);
		err.push_back(make_pair("CopyParamsToDevice", cudaMemcpy(h_paramlist, d_params, sizeof(Ty) * 5, cudaMemcpyDeviceToHost)));
		cout << " size " << sizeof(OptionData)*NofOpts << endl;
		OptionData* h_exodlist = (OptionData*)malloc(sizeof(OptionData)*NofOpts);
		err.push_back(make_pair("CopyOptionDataToHost", cudaMemcpy(h_exodlist, d_ODlist, sizeof(OptionData)*NofOpts, cudaMemcpyDeviceToHost))); //Check for data corruption on device
		ErrorCheck();
		for (int m = 0; m < NofOpts; m++)
		{
			for (int k = 0; k < NofStrikes; k++)
			{
				cout << h_exodlist[m].mat << " " << h_exodlist[m].SL[k] << " " << h_exodlist[m].OT[k] << endl;
			}
		}
	}

	Ty value(const Vector<Ty> &x)
	{
		vector<Ty> outputvec(NofStrikes*NofOpts, 0.0);
		for (int i = 0; i < 5; i++)
		{
			params[i] = x[i];
		}
		cout << itercount << " " << params[0] << " " << params[1] << " " << params[2] << " " << params[3] << " " << params[4] << endl;
		err.push_back(make_pair("CopyParamsToDeviceFromValueFun", cudaMemcpy(d_params, params, sizeof(Ty) * 5, cudaMemcpyHostToDevice)));
		ErrorCheck();
		HestonKernelWrapper(d_params, h_ODlist, d_ODlist, h_array, d_array, res_array, size);
		int optcounter = 0;
		Ty obsum = 0.;
		for (int m = 0; m < NofOpts; m++)
		{
			for (int k = 0; k < NofStrikes; k++)
			{
				obsum += pow(res_array[m].data[k] - h_ODlist[m].MPL[k], 2);
				optcounter += 1;
			}
		}
		itercount += 1;
		return obsum;		
	}

	vector<Ty> EnsemblePricer(vector<Ty> inputvec)
	{
		vector<Ty> outputvec(NofStrikes*NofOpts, 0.0);
		for (int i = 0; i < 5; i++)
		{
			params[i] = inputvec[i];
		}
		cout << itercount << " " << params[0] << " " << params[1] << " " << params[2] << " " << params[3] << " " << params[4] << endl;
		err.push_back(make_pair("CopyParamsToDeviceFromValueFun", cudaMemcpy(d_params, params, sizeof(Ty) * 5, cudaMemcpyHostToDevice)));
		ErrorCheck();
		HestonKernelWrapper(d_params, h_ODlist, d_ODlist, h_array, d_array, res_array, size);
		int optcounter = 0;

		for (int m = 0; m < NofOpts; m++)
		{
			for (int k = 0; k < NofStrikes; k++)
			{
				outputvec[optcounter] = res_array[m].data[k] - h_ODlist[m].MPL[k];
				optcounter += 1;
			}
		}
		itercount += 1;
		return outputvec;
	}

	void PricePrintout(vector<Ty> x)
	{
		for (int i = 0; i < 5; i++)
		{
			params[i] = x[i];
		}
		err.push_back(make_pair("CopyParamsToDeviceFromPricePrintout", cudaMemcpy(d_params, params, sizeof(Ty) * NofParams, cudaMemcpyHostToDevice)));
		HestonKernelWrapper(d_params, h_ODlist, d_ODlist, h_array, d_array, res_array, size);
		Ty sum = 0;
		for (int m = 0; m < NofOpts; m++)
		{
			for (int k = 0; k < NofStrikes; k++)
				cout << "price " << res_array[m].data[k] << " mid price " << h_ODlist[m].MPL[k] << endl;
		}
	}



private:
	int size;
	int ex_size;
	int NofFunCalls;
	int NofGradCalls;
	int itercount;
	Ty* params;
	OptionData* h_ODlist;
	OptionData* d_ODlist;
	Ty* d_params;
	Ty* h_array;
	Ty* d_array;
	OptionData* gh_ODlist;// = (OptionData*)malloc(sizeof(OptionData)*NofOpts);
	CudaVector res_array[NofOpts];
	
};