#ifndef _DEVICEAABB_CUH_
#define _DEVICEAABB_CUH_
#include "cudaheaders.cuh"

class DeviceAABB
{
public:
	float Bounds[6];
	__host__ __device__ DeviceAABB()
	{

	}

	__host__ __device__ void operator =(const DeviceAABB& Obj)
	{
		*(Bounds) = *(Obj.Bounds);
		*(Bounds + 1) = *(Obj.Bounds + 1);
		*(Bounds + 2) = *(Obj.Bounds + 2);
		*(Bounds + 3) = *(Obj.Bounds + 3);
		*(Bounds + 4) = *(Obj.Bounds + 4);
		*(Bounds + 5) = *(Obj.Bounds + 5);
	}

	__host__ __device__ bool operator ==(const DeviceAABB& Obj) const
	{
		return *(Bounds) == *(Obj.Bounds) &&
			*(Bounds + 1) == *(Obj.Bounds + 1) &&
			*(Bounds + 2) == *(Obj.Bounds + 2) &&
			*(Bounds + 3) == *(Obj.Bounds + 3) &&
			*(Bounds + 4) == *(Obj.Bounds + 4) &&
			*(Bounds + 5) == *(Obj.Bounds + 5);
	}


	__host__ __device__ DeviceAABB& operator +(const DeviceAABB& other) const
	{
		DeviceAABB result;

		*(result.Bounds) = fmin(*(Bounds), *(other.Bounds));
		*(result.Bounds + 1) = fmin(*(Bounds + 1), *(other.Bounds + 1));
		*(result.Bounds + 2) = fmin(*(Bounds + 2), *(other.Bounds + 2));
		*(result.Bounds + 3) = fmax(*(Bounds + 3), *(other.Bounds + 3));
		*(result.Bounds + 4) = fmax(*(Bounds + 4), *(other.Bounds + 4));
		*(result.Bounds + 5) = fmax(*(Bounds + 5), *(other.Bounds + 5));

		return result;
	}

	__host__ __device__
		void operator +=(const DeviceAABB& other)
	{
		*(this->Bounds) = fmin(*(this->Bounds), *(other.Bounds));
		*(this->Bounds + 1) = fmin(*(this->Bounds + 1), *(other.Bounds + 1));
		*(this->Bounds + 2) = fmin(*(this->Bounds + 2), *(other.Bounds + 2));
		*(this->Bounds + 3) = fmax(*(this->Bounds + 3), *(other.Bounds + 3));
		*(this->Bounds + 4) = fmax(*(this->Bounds + 4), *(other.Bounds + 4));
		*(this->Bounds + 5) = fmax(*(this->Bounds + 5), *(other.Bounds + 5));
	}

	__host__ __device__
		float GetSurfaceArea() const
	{
		float length, height, width;

		width = Bounds[3] - Bounds[0];
		height = Bounds[4] - Bounds[1];
		length = Bounds[5] - Bounds[2];

		// return the surface area formula
		return 2 * (length * width + length * height + width * height);
	}

	__host__ __device__
		float3 GetCentroid() const
	{
		float3 result;
		result.x = (Bounds[0] + Bounds[3]) * 0.5f;
		result.y = (Bounds[1] + Bounds[4]) * 0.5f;
		result.z = (Bounds[2] + Bounds[5]) * 0.5f;
		return result;
	}

	__host__ __device__
		void Assign(const DeviceAABB& first, const DeviceAABB& second)
	{
		*(this->Bounds) = fmin(*(first.Bounds), *(second.Bounds));
		*(this->Bounds + 1) = fmin(*(first.Bounds + 1), *(second.Bounds + 1));
		*(this->Bounds + 2) = fmin(*(first.Bounds + 2), *(second.Bounds + 2));
		*(this->Bounds + 3) = fmax(*(first.Bounds + 3), *(second.Bounds + 3));
		*(this->Bounds + 4) = fmax(*(first.Bounds + 4), *(second.Bounds + 4));
		*(this->Bounds + 5) = fmax(*(first.Bounds + 5), *(second.Bounds + 5));
	}

	__host__ __device__
		bool Collide(const DeviceAABB& otherBox) const
	{
		return Bounds[3] > otherBox.Bounds[0] && Bounds[0] < otherBox.Bounds[3] &&
			Bounds[4] > otherBox.Bounds[1] && Bounds[1] < otherBox.Bounds[4] &&
			Bounds[5] > otherBox.Bounds[2] && Bounds[2] < otherBox.Bounds[5];
	}

	__host__ __device__ float GetCombinedSurfaceArea(const DeviceAABB& other) const
	{
		float length, height, width;

		width = fmax(*(this->Bounds + 3), *(other.Bounds + 3)) - fmin(*(this->Bounds), *(other.Bounds));
		height = fmax(*(this->Bounds + 4), *(other.Bounds + 4)) - fmin(*(this->Bounds + 1), *(other.Bounds + 1));
		length = fmax(*(this->Bounds + 5), *(other.Bounds + 5)) - fmin(*(this->Bounds + 2), *(other.Bounds + 2));

		// return the surface area formula
		return 2 * (length * width + length * height + width * height);
	}

	__host__ __device__ float ComputeSAH(const DeviceAABB& Node2) const
	{
		return (this->GetSurfaceArea() + Node2.GetSurfaceArea()) / float((*this + Node2).GetSurfaceArea());
	}
};
#endif