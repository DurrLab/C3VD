/***********************************************************************************/
/*
 *	File name:	Mask.cpp
 *
 *	Author:     Taylor Bobrow, Johns Hopkins University (2023)
 * 
 */

#include "Mask.h"

#include <cuda_runtime_api.h>
#include <cuda.h>

Mask::Mask(pangolin::TypedImage mask)
:   width(mask.w),
    height(mask.h)
{
    /* Convert the image to a binary mask on the host. */
    bool* mask_host = (bool*)malloc(width*height*sizeof(bool));
    for(int y = 0; y < height; y++)
        for(int x = 0; x < width; x++)
            mask_host[y*height + x] = mask(x,y) > 0 ? 1 : 0;
    
    /* Move the binary mask to the device. */
    cudaMalloc((void**) &mask_dev, sizeof(bool)*width*height);
    cudaMemcpy(mask_dev, mask.ptr, sizeof(bool)*width*height, cudaMemcpyHostToDevice);

    free(mask_host);
}

Mask::~Mask(void)
{
    cudaFree(mask_dev);
}

template <typename T>
__global__ void apply_kernel(T *img_dev, bool *mask_dev, const unsigned int width, const unsigned int height)
{
    int pixelX = threadIdx.x + blockIdx.x*blockDim.x;
    int pixelY = threadIdx.y + blockIdx.y*blockDim.y;
    if (pixelX >= width) return;
    if (pixelY >= height) return;

    if(mask_dev[pixelX + width*pixelY])
        return;
    
    img_dev[pixelX + width*pixelY] = 0;

    return;
}

template <typename T>
void Mask::apply(T *img_dev)
{
    const dim3 blockSize(32, 32, 1);

    const dim3 gridSize(width/blockSize.x + 1, height/blockSize.y + 1, 1);

    apply_kernel<<<gridSize,blockSize>>>(img_dev, mask_dev, width, height);

    cudaDeviceSynchronize();
}

template void Mask::apply(uint8_t  *img_dev);
template void Mask::apply(uint16_t *img_dev);
template void Mask::apply(uint32_t *img_dev);
template void Mask::apply(uint64_t *img_dev);