#ifndef _CUDAHEADERS_H_
#define _CUDAHEADERS_H_
#include <cuda.h>
#include <cuda_runtime.h>
typedef unsigned int uint;

template<typename T> void GetMaximumOccupancyForFunction(int &gridSize, int &blockSize, uint Size, T func)
{
	//blockSize;   // The launch configurator returned block size 
	int minGridSize; // The minimum grid size needed to achieve the 
	// maximum occupancy for a full device launch 
	//gridSize;    // The actual grid size needed, based on input size 

	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, func, 0, 0);

	if (blockSize != 0)
		// Round up according to array size 
		gridSize = (Size + blockSize - 1) / blockSize;
	else
		gridSize = minGridSize;
}
#endif