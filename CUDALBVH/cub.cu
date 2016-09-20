#include "DeviceFunctions.cuh"
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh> 



namespace CUBLIB
{
	void Sort(uint*& keys, uint*& objectsIDs, uint size)
	{
		/// Allocate the neccesary 
		uint* sortedKeys;
		cudaMalloc(&sortedKeys, sizeof(uint) * size);
		uint *sortedIDs;
		cudaMalloc(&sortedIDs, sizeof(uint) * size);

		// Allocate temporary storage for sorting
		size_t  temp_storage_bytes = 0;
		void    *d_temp_storage = NULL;
		// Find the temp storage
		cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, keys, sortedKeys, objectsIDs, sortedIDs, size);
		// Allocate temporary storage
		cudaMalloc(&d_temp_storage, temp_storage_bytes);
		// Start the sort
		cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, keys, sortedKeys, objectsIDs, sortedIDs, size);


		cudaFree(keys);
		cudaFree(objectsIDs);

		objectsIDs = sortedIDs;
		keys = sortedKeys;
	}


	uint* PrefixSum(uint* keys, uint size)
	{
		uint* keys_out;
		cudaMalloc(&keys_out, sizeof(uint) * size);
		// Allocate temporary storage
		size_t  temp_storage_bytes = 0;
		void    *d_temp_storage = NULL;
		//// Find the temp storage
		cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, keys, keys_out, size);
		// Allocate temporary storage for inclusive prefix sum
		cudaMalloc(&d_temp_storage, temp_storage_bytes);
		// Run inclusive prefix sum
		cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, keys, keys_out, size);

		return keys_out;
	}
}

