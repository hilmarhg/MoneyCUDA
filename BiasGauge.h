#include <iostream>
#include <random>
#include <math.h>
#include <vector>
#include "Globals.h"

class BiasGauge
{
public:
	static Ty Mean(const Ty* arr, const int asize)
	{
		Ty sum = 0;
		for (int i = 0; i < asize; i++)
		{
			sum += arr[i];
		}
		return sum / (Ty)asize;
	}

	static Ty Variance(const Ty* arr, const int asize)
	{
		Ty sum = 0;
		Ty mean = Mean(arr, asize);
		for (int i = 0; i < asize; i++)
		{
			sum += (arr[i] - mean)*(arr[i] - mean);
		}
		return sum / (Ty)(asize - 1);
	}

	static Ty Covariance(const Ty* arr1, const Ty* arr2, const int asize)
	{
		Ty sum = 0;
		Ty mean1 = Mean(arr1, asize);
		Ty mean2 = Mean(arr2, asize);
		Ty var1 = Variance(arr1, asize);
		Ty var2 = Variance(arr2, asize);
		for (int i = 0; i < asize; i++)
		{
			sum += (arr1[i] - mean1)*(arr2[i] - mean2);
		}
		Ty cov = sum / (Ty)(asize - 1);
		return  cov;
	}

	static Ty Correlation(const Ty* arr1, const Ty* arr2, const int asize)
	{
		Ty sum = 0;
		Ty mean1 = Mean(arr1, asize);
		Ty mean2 = Mean(arr2, asize);
		Ty var1 = Variance(arr1, asize);
		Ty var2 = Variance(arr2, asize);
		for (int i = 0; i < asize; i++)
		{
			sum += (arr1[i] - mean1)*(arr2[i] - mean2);
		}
		Ty cov = sum / (Ty)(asize - 1);
		return  cov / sqrt(var1*var2);
	}

	static Ty KolmogorovSmirnov(const std::vector<Ty> EmpiricalSample, const std::vector<Ty> ReferenceSample)
	{
		//Distribution normality check, under construction.
	}
};

