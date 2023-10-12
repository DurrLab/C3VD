/***********************************************************************************/
/*
 *	File name:	Rgba2rgb.cu
 *
 *	Author:     Taylor Bobrow, Johns Hopkins University (2023)
 * 
 */

#include "Rgba2rgb.cuh"

#include <cuda.h>
#include <cuda_runtime_api.h>

template <typename T>
__global__ void rgba2rgb_kernel(T *rgba_dev,
                                T *rgb_dev,
                                const unsigned int width,
                                const unsigned int height)
{
    int pixelX = threadIdx.x + blockIdx.x*blockDim.x;
    int pixelY = threadIdx.y + blockIdx.y*blockDim.y;
    if (pixelX >= width) return;
    if (pixelY >= height) return;

    for (int chnl = 0; chnl<3; chnl++)
        rgb_dev[pixelY*width*3 + pixelX*3 + chnl] = rgba_dev[pixelY*width*4 + pixelX*4 + chnl];

    return;
}

template <typename T>
void rgba2Rgb(T *rgba_dev,
              T *rgb_dev,
              const unsigned int width,
              const unsigned int height)
{
    const dim3 blockSize(32, 32, 1);

    const dim3 gridSize(width/blockSize.x + 1, height/blockSize.y + 1, 1);

    rgba2rgb_kernel<<<gridSize,blockSize>>>(rgba_dev, rgb_dev, width, height);

    cudaDeviceSynchronize();
}

template void rgba2Rgb(uint8_t  *rgba_dev, uint8_t  *rgb_dev, const unsigned int width, const unsigned int height);
template void rgba2Rgb(uint16_t *rgba_dev, uint16_t *rgb_dev, const unsigned int width, const unsigned int height);
template void rgba2Rgb(uint32_t *rgba_dev, uint32_t *rgb_dev, const unsigned int width, const unsigned int height);