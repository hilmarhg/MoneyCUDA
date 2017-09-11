#pragma once
#include <iostream>
#include <vector>
#include <string>
#include <cuda_runtime.h>

class CmdlDebug
{
public:
	void ErrorCheck()
	{
		std::string sin;
		for (auto p = err.begin(); p < err.end(); p++)
		{
			if (p->second != 0)
			{
				std::cout << "Execution failed at " << p->first << " , Continue Y/N?";
				std::cin >> sin;
				if (sin.compare("N"))
					exit;
			}
		}
	}

	std::vector<std::pair<std::string, cudaError_t>> err;
};