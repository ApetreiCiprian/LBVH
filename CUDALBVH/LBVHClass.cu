#include "LBVHClass.h"
#include "DeviceFunctions.cuh"

namespace DEVICECUDALBVH
{
	LBVHClass::LBVHClass()
	{
		_initialized = false;
	}

	LBVHClass::~LBVHClass()
	{
		if (_initialized)
		{
			ClearBVH();
		}
	}


	void LBVHClass::ClearBVH()
	{
		cudaFree(hostBVH->InternalNodesAABBs);
		cudaFree(hostBVH->InternalNodesChildren);
		cudaFree(hostBVH->LeafNodesAABBs);
		cudaFree(hostBVH->RangeOfKeys);
		cudaFree(hostBVH->SortedObjectsIDs);
		cudaFree(Bvh);
	}



	void LBVHClass::ConstructHierarchy(void* AABBs, uint arraySize, float MinValues[3], float MaxValues[3])
	{
		Size = arraySize;

		float3 MinCoord;
		MinCoord.x = MinValues[0];
		MinCoord.y = MinValues[1];
		MinCoord.z = MinValues[2];

		float3 MaxCoord;
		MaxCoord.x = MaxValues[0];
		MaxCoord.y = MaxValues[1];
		MaxCoord.z = MaxValues[2];

		hostBVH = new DeviceBVH();
		hostBVH->Size = arraySize;

		cudaMalloc(&Bvh, sizeof(DeviceBVH));
		cudaMalloc(&(hostBVH->LeafNodesAABBs), sizeof(DeviceAABB) * Size);
		cudaMemcpy(hostBVH->LeafNodesAABBs, AABBs, Size * sizeof(DeviceAABB), cudaMemcpyKind::cudaMemcpyHostToDevice);
		cudaMalloc(&(hostBVH->InternalNodesAABBs), sizeof(DeviceAABB) * Size);
		cudaMalloc(&(hostBVH->RangeOfKeys), sizeof(uint2) * Size);
		cudaMalloc(&(hostBVH->InternalNodesChildren), sizeof(uint2) * Size);
		cudaMalloc(&(hostBVH->SortedObjectsIDs), sizeof(uint) * Size);
		_initialized = true;

		uint* mortonCodes = CUDALBVH::ComputeMortonKeys(hostBVH->LeafNodesAABBs, MinCoord, MaxCoord, Size);
		CUDALBVH::SequenceArray(hostBVH->SortedObjectsIDs, Size);
		CUBLIB::Sort(mortonCodes, hostBVH->SortedObjectsIDs, Size);
		CUDALBVH::AABBPermuntation(hostBVH->LeafNodesAABBs, hostBVH->SortedObjectsIDs, Size);

		cudaMemcpy(Bvh, hostBVH, sizeof(DeviceBVH), cudaMemcpyKind::cudaMemcpyHostToDevice);

		CUDALBVH::GenerateMortonBVH(mortonCodes, Bvh, Size);
	}

	LBVHCollisionPairs& LBVHClass::GetCollisionPairs(uint CollisionListMaxSize)
	{
		LBVHCollisionPairs* test = new LBVHCollisionPairs();
		test->PairsIndex = CUDALBVH::GetCollisionList(CollisionListMaxSize, Bvh, Size, true);
		test->Size = CollisionListMaxSize;
		ClearBVH();
		return *test;

	}


}
