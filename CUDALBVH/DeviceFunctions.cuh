#include "DeviceBVH.cuh"
#include "TraversalResult.h"

namespace CUDALBVH
{
	void GenerateMortonBVH(uint* &mortonCodes, DeviceBVH* &bvh, uint size);

	uint* ComputeMortonKeys(DeviceAABB*& objectsAABBs, float3 MinCoord, float3 MaxCoord, uint size);

	DEVICECUDALBVH::CollisionPair* GetCollisionList(uint& collisionArraySize, DeviceBVH* &bvh, uint size, bool stackless = true);

	void SequenceArray(uint*& input, uint size);

	void DefaultValueArray(uint*& input, uint value, uint size);

	void AABBPermuntation(DeviceAABB*& inputAABBs, uint*& permutationIDs, uint size);

	cudaError_t checkCuda(cudaError_t result);

}


namespace CUBLIB
{
	void Sort(uint*& keys, uint*& objectsIDs, uint size);

	uint* PrefixSum(uint* keys, uint size);
}
