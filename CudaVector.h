#pragma once
#include "Globals.h"

class CudaVector
{
public:
	CUDA_CALLABLE_MEMBER CudaVector()
	{
		for (int i = 0; i < NofStrikes; i++)
		{
			data[i] = (Ty)0;
		}
	}

	CUDA_CALLABLE_MEMBER CudaVector(Ty vec[])
	{
		for (int i = 0; i < NofStrikes; i++)
		{
			data[i] = vec[i];
		}
	}
	CUDA_CALLABLE_MEMBER CudaVector& operator+=(const Ty a1[])
	{
		for (int i = 0; i < NofStrikes; i++)
		{
			data[i] += a1[i];
		}

		return *this;
	}

	CUDA_CALLABLE_MEMBER CudaVector& operator+=(const CudaVector a1)
	{
		for (int i = 0; i < NofStrikes; i++)
		{
			data[i] += a1.data[i];
		}

		return *this;
	}

	CUDA_CALLABLE_MEMBER CudaVector& operator-=(Ty a1)
	{
		for (int i = 0; i < NofStrikes; i++)
		{
			data[i] -= a1;
		}

		return *this;
	}

	CUDA_CALLABLE_MEMBER CudaVector& operator/(Ty denominator)
	{
		for (int i = 0; i < NofStrikes; i++)
		{
			data[i] /= denominator;
		}
		return *this;
	}

	//static const int vecsize = NofStrikes;

	Ty data[NofStrikes];
};
