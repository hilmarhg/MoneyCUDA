# MoneyCUDA
Derivative Pricing Algorithms with CUDA

The code in this repository revolves around applying GPU programming to solving the nonlinear least squares calibration problem
of derivative pricing models which do not exhibit a closed form solution. These models are normally given as a system of stochastic
differential equations (SDEs), and the mathematical expectation of the processes the SDEs describe in this implementation is computed 
using Monte Carlo simulation. They are calibrated against market prices of vanilla derivative contracts.

The first test case implementation is for Heston's stochastic volatility model, see https://en.wikipedia.org/wiki/Heston_model. Later on,
I will add more complex models, with more stochastic factors. In addition, I am adding characteristic functions of different affine 
models to the solution since they often prove to be effective control variates for the simulation of more complex models (The 
Heston model is in fact one of those as well).

The nonlinear optimization routine used in this solution is the BFGS algorithm implemented in https://github.com/PatWie/CppNumericalSolvers

Currently, an overview of the solution is as follows:

Main.cpp: The entry point of the program, instantiates the ModelCalibration class and calls its member functions

ModelCalibration.h: The class that sets up the calibration problem and calls the nonlinear optimization routine.

ParallelizedObjective.h: The class that contains the objective function, computed using the GPU.

SimulationKernels.h: The header file for the SimulationKernel.cu file. 

SimulationKernels.cu: The implementation of the GPU functions employed by the ParallelizedObjective class. Note that the kernels are not
available as member functions of a class because there is no state to attach them to, as per the current CUDA architecture.

CudaVector.h: A datatype that works on the host as well as the device. It will get more linear algebra specific functionality 
in the future.

OptionData.h: A container for the option data we use for the calibration.

BiasGauge.h: A class of static methods for checking whether the random numbers generated in the MC simulation have the desired 
sample properties.

CmdlDebug: A class that contains a simple method for keeping track of CUDA related errors.

FourierKernels.cu: Currently empty but will hold the characteristic functions of the affine models that will be used as control variates
for the more complex models

