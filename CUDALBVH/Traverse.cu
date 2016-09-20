#include "DeviceFunctions.cuh"

__global__ void TraverseStackless(const DeviceBVH* bvh, uint2* list, uint* listSize, const uint size, const uint listCapacity)
{
	uint i = threadIdx.x + blockIdx.x * blockDim.x;
	while (i < size)
	{
		uint currLeaf = i;
		DeviceAABB& queryAABB = bvh->LeafNodesAABBs[currLeaf];
		bool collides;
		bool traverseRightChild = true;

		uint curr = bvh->getDfsNextNode(currLeaf ^ IS_LEAF);
		// Start the collision detection
		while (curr < size - 1)
		{
			curr = (traverseRightChild) ? bvh->InternalNodesChildren[curr].y : bvh->InternalNodesChildren[curr].x;

			collides = queryAABB.Collide(bvh->GetNodeAABB(curr));

			if (collides)
			{
				if (curr & IS_LEAF)
				{
					uint index = atomicInc(listSize, listCapacity);
					list[index].x = bvh->GetBaseObjectIdx(currLeaf);
					list[index].y = bvh->GetBaseObjectIdx(curr ^ IS_LEAF);

				}
				else
				{
					traverseRightChild = false;
					continue;
				}
			}

			curr = bvh->getDfsNextNode(curr);
			traverseRightChild = true;
		}


		i += blockDim.x * gridDim.x;
	}
}

__global__ void Initialize(uint value, uint* dest)
{
	dest[0] = value;
}

namespace CUDALBVH
{

	DEVICECUDALBVH::CollisionPair* GetCollisionList(uint& collisionArraySize, DeviceBVH*& bvh, uint size, bool stackless)
	{
		int blockSize = 0;   // The launch configurator returned block size 
		int gridSize = 0;    // The actual grid size needed, based on input size 

		GetMaximumOccupancyForFunction(gridSize, blockSize, size, TraverseStackless);


		uint2* deviceCollisionList;
		cudaMalloc(&deviceCollisionList, sizeof(uint2) * collisionArraySize);
		uint* collisionListSize;
		cudaMalloc(&collisionListSize, sizeof(uint));
		Initialize << <1, 1 >> >(0, collisionListSize);


		TraverseStackless << <gridSize, blockSize >> > (bvh, deviceCollisionList, collisionListSize, size, collisionArraySize);

		cudaMemcpy(&collisionArraySize, collisionListSize, sizeof(uint), cudaMemcpyKind::cudaMemcpyDeviceToHost);
		DEVICECUDALBVH::CollisionPair* hostArray = new DEVICECUDALBVH::CollisionPair[collisionArraySize];
		cudaMemcpy(&hostArray[0], deviceCollisionList, sizeof(uint2) * collisionArraySize, cudaMemcpyKind::cudaMemcpyDeviceToHost);

		cudaFree(collisionListSize);
		cudaFree(deviceCollisionList);

		return &hostArray[0];
	}

}
