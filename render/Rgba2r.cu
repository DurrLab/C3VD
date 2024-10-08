/***********************************************************************************/
/*
 *	File name:	Rgba2rgb.cu
 *
 *	Author:     Taylor Bobrow, Johns Hopkins University (2023)
 * 
 */

#include "Rgba2r.cuh"

#include <cuda.h>
#include <cuda_runtime_api.h>

template <typename T>
__global__ void rgba2R_kernel(T *rgba_dev,
                              T *r_dev,
                              const unsigned int width,
                              const unsigned int height)
{
    int pixelX = threadIdx.x + blockIdx.x*blockDim.x;
    int pixelY = threadIdx.y + blockIdx.y*blockDim.y;
    if (pixelX >= width) return;
    if (pixelY >= height) return;

    r_dev[pixelY*width + pixelX] = rgba_dev[pixelY*width*4 + pixelX*4];

    return;
}

template <typename T>
void rgba2R(T *rgba_dev,
            T *r_dev,
            const unsigned int width,
            const unsigned int height)
{
    const dim3 blockSize(32, 32, 1);

    const dim3 gridSize(width/blockSize.x + 1, height/blockSize.y + 1, 1);

    rgba2R_kernel<<<gridSize,blockSize>>>(rgba_dev, r_dev, width, height);

    cudaDeviceSynchronize();
}

template void rgba2R(uint8_t  *rgba_dev, uint8_t  *r_dev, const unsigned int width, const unsigned int height);
template void rgba2R(uint16_t *rgba_dev, uint16_t *r_dev, const unsigned int width, const unsigned int height);
template void rgba2R(uint32_t *rgba_dev, uint32_t *r_dev, const unsigned int width, const unsigned int height);