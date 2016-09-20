#include "DeviceFunctions.cuh"
#include "stdio.h"

__global__ void SequenceArrayDevice(uint* __restrict__ InputArray, const uint size)
{
	uint i = threadIdx.x + blockIdx.x * blockDim.x;
	while (i < size)
	{
		InputArray[i] = i;
		i += blockDim.x * gridDim.x;
	}
}

__global__ void DefaultValueArrayDevice(uint* __restrict__ InputArray, const uint value, const uint size)
{
	uint i = threadIdx.x + blockIdx.x * blockDim.x;
	while (i < size)
	{
		InputArray[i] = value;
		i += blockDim.x * gridDim.x;
	}
}


__global__ void Permutation(const DeviceAABB* __restrict__ Input, DeviceAABB* __restrict__ Output, const uint* __restrict__ permutationIndexes, const uint size)
{
	uint i = threadIdx.x + blockIdx.x * blockDim.x;
	while (i < size)
	{
		Output[i] = Input[permutationIndexes[i]];
		i += blockDim.x * gridDim.x;
	}
}

namespace CUDALBVH
{
	void AABBPermuntation(DeviceAABB*& inputAABBs, uint*& permutationIDs, uint size)
	{
		DeviceAABB* orderedAABBs;
		cudaMalloc(&orderedAABBs, sizeof(DeviceAABB) * size);

		int blockSize = 0;   // The launch configurator returned block size 
		int gridSize = 0;    // The actual grid size needed, based on input size 
		GetMaximumOccupancyForFunction(gridSize, blockSize, size, Permutation);

		Permutation << <gridSize, blockSize >> > (inputAABBs, orderedAABBs, permutationIDs, size);

		cudaFree(inputAABBs);
		inputAABBs = orderedAABBs;
	}
	void AABBPermuntation(DeviceAABB*& inputAABBs, DeviceAABB*& outputAABBs, uint*& permutationIDs, uint size)
	{
		int blockSize = 0;   // The launch configurator returned block size 
		int gridSize = 0;    // The actual grid size needed, based on input size 
		GetMaximumOccupancyForFunction(gridSize, blockSize, size, Permutation);

		Permutation << <gridSize, blockSize >> > (inputAABBs, outputAABBs, permutationIDs, size);
	}
	void SequenceArray(uint*& input, uint size)
	{
		int blockSize = 0;   // The launch configurator returned block size 
		int gridSize = 0;    // The actual grid size needed, based on input size 
		GetMaximumOccupancyForFunction(gridSize, blockSize, size, SequenceArrayDevice);

		// Initialize an array with a sequence representing the id of the each of the input DeviceAABB
		SequenceArrayDevice << <gridSize, blockSize >> > (input, size);
	}

	void DefaultValueArray(uint*& input, uint value, uint size)
	{
		int blockSize = 0;   // The launch configurator returned block size 
		int gridSize = 0;    // The actual grid size needed, based on input size 
		GetMaximumOccupancyForFunction(gridSize, blockSize, size, DefaultValueArrayDevice);

		// Initialize an array with a sequence representing the id of the each of the input DeviceAABB
		DefaultValueArrayDevice << <gridSize, blockSize >> > (input, value, size);
	}

	cudaError_t checkCuda(cudaError_t result)
	{
#if defined(DEBUG) || defined(_DEBUG)
		if (result != cudaSuccess) {
			fprintf(stderr, "CUDA Runtime Error: %sn", cudaGetErrorString(result));
		}
#endif
		return result;
	}
}
