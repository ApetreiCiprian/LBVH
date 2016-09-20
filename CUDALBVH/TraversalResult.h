#pragma once
typedef unsigned int uint;

namespace DEVICECUDALBVH
{
	struct CollisionPair
	{
		uint x, y;
	};

	struct LBVHCollisionPairs
	{
		CollisionPair* PairsIndex;
		uint Size;
	};

}
