#pragma once
#include "TraversalResult.h"
#include "DeviceBVH.cuh"
namespace DEVICECUDALBVH
{

	class LBVHClass
	{
	public:
		LBVHClass();
		~LBVHClass();
		void ConstructHierarchy(void* AABBs, uint arraySize, float MinCoords[3], float MaxCoords[3]);
		LBVHCollisionPairs& GetCollisionPairs(uint CollisionListMaxSize);

		void ClearBVH();

		uint Size;

	private:

		DeviceBVH* hostBVH, *Bvh;
		bool _initialized;
	};
}
