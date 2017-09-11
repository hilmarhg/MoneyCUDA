#pragma once
#include <iostream>
#include "Globals.h"

class OptionData
{
public:
	CUDA_CALLABLE_MEMBER OptionData(Ty underlying, Ty maturity, Ty rate, Ty* sl, Ty* mpl, Ty* ot)
	{
		ap = underlying;
		mat = maturity;
		rf = rate;
		for (int i = 0; i < NofStrikes; i++)
		{
			SL[i] = sl[i];
			MPL[i] = mpl[i];
			OT[i] = ot[i];
		}
	}

	CUDA_CALLABLE_MEMBER OptionData()
	{

	}

	void Printout()
	{
		for (int i = 0; i < NofStrikes; i++)
		{
			std::cout << "mat " << mat << " strike " << SL[i] << " " << "otype " << OT[i] << std::endl;
		}
	}

	Ty ap;
	Ty mat;
	Ty rf;
	Ty SL[NofStrikes];
	Ty OT[NofStrikes];
	Ty MPL[NofStrikes];
	Ty BIP[NofStrikes];
	Ty ASP[NofStrikes];
	Ty spotvol;

};