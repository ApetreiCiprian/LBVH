#pragma once
#include "DeviceAABB.cuh"
#define IS_LEAF 2147483648

class DeviceBVH
{
public:
	__host__ __device__ DeviceBVH()
	{
		Size = UINT_MAX;
		Root = UINT_MAX;
		Padding = UINT_MAX;
	}
	///===========================================================
	///					Device Functions
	///===========================================================
	__device__ bool IsLeaf(uint currentIndex) const
	{
		return (currentIndex & IS_LEAF) != 0;
	}

	__host__ __device__ void SetNodeBoundingBoxFromChildren(uint currentNodeIndex) const
	{
		InternalNodesAABBs[currentNodeIndex].Assign(GetNodeAABB(InternalNodesChildren[currentNodeIndex].x), GetNodeAABB(InternalNodesChildren[currentNodeIndex].y));
	}

	__host__ __device__ DeviceAABB& GetNodeAABB(const uint currentNodeIndex) const
	{
		if (currentNodeIndex & IS_LEAF)
		{
			return LeafNodesAABBs[currentNodeIndex ^ IS_LEAF];
		}
		else
		{
			return InternalNodesAABBs[currentNodeIndex];
		}
	}

	__host__ __device__ uint GetLeafArrayIndex(uint index) const
	{
		return index ^ IS_LEAF;
	}

	__host__ __device__ uint GetBaseObjectIdx(uint leafIndex) const
	{
		return SortedObjectsIDs[leafIndex & (~IS_LEAF)];
	}

	__host__ __device__ uint getDfsNextNode(uint currNode) const
	{
		if (currNode & IS_LEAF)
			return currNode ^ IS_LEAF;
		else
			return RangeOfKeys[currNode].y;
	}

	__host__ __device__ uint GetRoot()
	{
		if (Root != UINT_MAX)
		{
			return Root;
		}
		else
		{
			return Root = InternalNodesChildren[Size - 1].x;
		}
	}


	__host__ __device__ uint GetRoot() const
	{
		return InternalNodesChildren[Size - 1].x;
	}

	__host__ __device__ uint GetRange(uint nodeIdx) const
	{
		if (nodeIdx & IS_LEAF)
			return 1;

		return RangeOfKeys[nodeIdx].y - RangeOfKeys[nodeIdx].x;
	}

	__host__ __device__ uint2 GetRangeOfKeys(uint nodeIdx) const
	{
		uint2 parents;
		if (nodeIdx & IS_LEAF)
		{
			parents.y = nodeIdx ^ IS_LEAF;
			parents.x = parents.y - 1;
		}
		else
		{
			parents = RangeOfKeys[nodeIdx];
		}
		return parents;
	}

	__host__ __device__ uint GetParent(uint nodeIdx) const
	{
		uint2 parents = GetRangeOfKeys(nodeIdx);
		if (parents.x == -1 || RangeOfKeys[parents.x].y != parents.y)
			return parents.y;
		return parents.x;
	}

	__host__ __device__ uint GetSibling(uint nodeIdx) const
	{
		uint2 parents = GetRangeOfKeys(nodeIdx);
		if (parents.x == -1 || RangeOfKeys[parents.x].y != parents.y)
			return InternalNodesChildren[parents.y].y;
		return InternalNodesChildren[parents.x].x;
	}


	__host__ __device__ bool EvaluateNodePosition(uint nodeIdx) const
	{
		// if it's the first or last leaf, return true
		uint2 RangeOfKeys = GetRangeOfKeys(nodeIdx);
		if (RangeOfKeys.x == -1 || RangeOfKeys.y == Size - 1)
			return true;
		uint ancestor;
		if (InternalNodesChildren[RangeOfKeys.x].y != nodeIdx)
			ancestor = RangeOfKeys.x; // parent
		else
			ancestor = RangeOfKeys.y;

		// Test if the ancestor is 
		return InternalNodesAABBs[ancestor].GetSurfaceArea() <
			GetNodeAABB(InternalNodesChildren[RangeOfKeys.x].x).GetCombinedSurfaceArea(GetNodeAABB(InternalNodesChildren[RangeOfKeys.y].y));
	}

	///===========================================================
	///					Members
	///===========================================================
	uint Size;
	uint Root;
	// Device arrays
	DeviceAABB* LeafNodesAABBs;
	DeviceAABB* InternalNodesAABBs;

	//bool* _isLeafNodeAwake;
	//bool* _isInternalNodeAwake;

	uint* SortedObjectsIDs;
	uint2* InternalNodesChildren;

	// For each node there are two possible parents that are located at the ends of the range of keys covered by this node
	// We choose the best parent one based on a distance function
	// The Right RangeOfKeys is a dfs skip list (contains the next inorder traversal for each node in the hierarchy)
	uint2* RangeOfKeys;
	uint Padding;
};