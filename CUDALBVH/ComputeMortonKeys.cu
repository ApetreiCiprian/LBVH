#include "DeviceFunctions.cuh"

__device__  uint expandBits(uint v)
{
	v = (v * 0x00010001u) & 0xFF0000FFu;
	v = (v * 0x00000101u) & 0x0F00F00Fu;
	v = (v * 0x00000011u) & 0xC30C30C3u;
	v = (v * 0x00000005u) & 0x49249249u;
	return v;
}

//  Calculates a 30-bit Morton code for the
// given 3D point located within the unit cube [0,1].
__device__ uint morton3D(const float x, const float y, const float z)
{
	uint xx = expandBits((uint)fmin(fmax(x * 1024.0f, 0.0f), 1023.0f));
	uint yy = expandBits((uint)fmin(fmax(y * 1024.0f, 0.0f), 1023.0f));
	uint zz = expandBits((uint)fmin(fmax(z * 1024.0f, 0.0f), 1023.0f));
	return xx * 4 + yy * 2 + zz;
}

__global__ void CalculateMortonCodes(const DeviceAABB* __restrict__ Objects, uint* __restrict__ MortonCodes, const float3 MinCoord, const float3 Denominator, const uint size)
{
	uint i = threadIdx.x + blockIdx.x * blockDim.x;

	while (i < size)
	{
		float Centroid0, Centroid1, Centroid2;
		Centroid0 = ((*(Objects + i)).Bounds[0] + (*(Objects + i)).Bounds[3]) * 0.5f;
		Centroid1 = ((*(Objects + i)).Bounds[1] + (*(Objects + i)).Bounds[4]) * 0.5f;
		Centroid2 = ((*(Objects + i)).Bounds[2] + (*(Objects + i)).Bounds[5]) * 0.5f;
		// We map the centroid to the unit cube and compute the Morton code
		MortonCodes[i] = morton3D((Centroid0 - MinCoord.x) * Denominator.x, (Centroid1 - MinCoord.y) * Denominator.y, (Centroid2 - MinCoord.z) * Denominator.z);
		i += blockDim.x * gridDim.x;
	}
}

__global__  void CalculateMaxExtents(const DeviceAABB* __restrict__ ObjectsAABBs, float3 MinCoord, float3 MaxCoord, uint size)
{
	uint i = threadIdx.x + blockIdx.x * blockDim.x;

	while (i < size)
	{
		MinCoord.x = fmin(MinCoord.x, ObjectsAABBs[i].Bounds[0]);
		MinCoord.y = fmin(MinCoord.y, ObjectsAABBs[i].Bounds[1]);
		MinCoord.z = fmin(MinCoord.z, ObjectsAABBs[i].Bounds[2]);
		MaxCoord.x = fmax(MaxCoord.x, ObjectsAABBs[i].Bounds[3]);
		MaxCoord.y = fmax(MaxCoord.y, ObjectsAABBs[i].Bounds[4]);
		MaxCoord.z = fmax(MaxCoord.z, ObjectsAABBs[i].Bounds[5]);
		i += blockDim.x * gridDim.x;
	}
}
namespace CUDALBVH
{
	uint* ComputeMortonKeys(DeviceAABB*& objectsAABBs, float3 MinCoord, float3 MaxCoord, uint size)
	{
		int blockSize = 0;   // The launch configurator returned block size 
		int gridSize = 0;    // The actual grid size needed, based on input size 


		uint* resultKeys;
		cudaMalloc(&resultKeys, sizeof(uint) * size);

		float3 Denominator;
		/// precalculate so that we don't do the operation for each thread
		Denominator.x = 1.0f / ((MaxCoord.x - MinCoord.x) + (MaxCoord.x == MinCoord.x));
		Denominator.y = 1.0f / ((MaxCoord.y - MinCoord.y) + (MaxCoord.y == MinCoord.y));
		Denominator.z = 1.0f / ((MaxCoord.z - MinCoord.z) + (MaxCoord.z == MinCoord.z));


		GetMaximumOccupancyForFunction(gridSize, blockSize, size, CalculateMortonCodes);
		CalculateMortonCodes << <gridSize, blockSize >> >
			(objectsAABBs, resultKeys, MinCoord, Denominator, size);

		return resultKeys;
	}
}
