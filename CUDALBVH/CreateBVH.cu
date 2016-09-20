#include "DeviceBVH.cuh"
#include "DeviceFunctions.cuh"

__device__ uint findParent(int leftParent, int rightParent, uint currentNode, const uint* __restrict__ splitPointsValues, DeviceBVH* __restrict__ bvh, uint size)
{
	// I use this in one big if so we don't have to check every time if i<0 or i<e.size
	if ((leftParent == -1) || ((rightParent < size - 1) &&
		(splitPointsValues[rightParent] <= splitPointsValues[leftParent])))
	{
		// Right Node is the parent
		bvh->InternalNodesChildren[rightParent].x = currentNode;
		// Pass to other possible parent upwards
		bvh->RangeOfKeys[rightParent].x = leftParent;
		return rightParent;
	}
	else
	{
		// Left Node is the parent
		bvh->InternalNodesChildren[leftParent].y = currentNode;
		// Pass to other possible parent upwards
		bvh->RangeOfKeys[leftParent].y = rightParent;
		return leftParent;
	}
}


__global__ void GenerateHierarchy(const uint* __restrict__ splitPointsValues, DeviceBVH* __restrict__ bvh, uint* Atomics, uint size)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	while (i < size)
	{
		uint curr;
		curr = findParent(i - 1, i, i | IS_LEAF, splitPointsValues, bvh, size);
		while (atomicXor(&Atomics[curr], 1))
		{
			bvh->SetNodeBoundingBoxFromChildren(curr);
			curr = findParent(bvh->RangeOfKeys[curr].x, bvh->RangeOfKeys[curr].y, curr, splitPointsValues, bvh, size);
		}
		i += blockDim.x * gridDim.x;
	}
}


__global__ void SetupMortonHierarchy(const uint* __restrict__ mortonCodes, uint* __restrict__ distances, uint* __restrict__ atomics, uint size)
{
	uint i = threadIdx.x + blockIdx.x * blockDim.x;
	while (i < size)
	{
		atomics[i] = 0;
		if (i < size - 1)
		{
			distances[i] = mortonCodes[i] ^ mortonCodes[i + 1];
		}
		i += blockDim.x * gridDim.x;
	}
}


namespace CUDALBVH
{

	void GenerateMortonBVH(uint* &mortonCodes, DeviceBVH* &bvh, uint size)
	{
		int blockSize;   // The launch configurator returned block size 
		int gridSize;    // The actual grid size needed, based on input size 
		GetMaximumOccupancyForFunction(gridSize, blockSize, size, SetupMortonHierarchy);
		uint* distances;
		uint* atomics;
		cudaMalloc(&atomics, sizeof(uint) * size);
		cudaMalloc(&distances, sizeof(uint) * size);
		SetupMortonHierarchy << <gridSize, blockSize >> > (mortonCodes, distances, atomics, size);

		GenerateHierarchy << <gridSize, blockSize >> > (distances, bvh, atomics, size);

		cudaFree(mortonCodes);
		cudaFree(distances);
		cudaFree(atomics);
	}
}
