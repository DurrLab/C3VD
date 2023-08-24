/***********************************************************************************/
/*
 *	File name:	LaunchParams.h
 *
 *	Author:     Taylor Bobrow, Johns Hopkins University (2023)
 * 
 */

#ifndef LAUNCHPARAMS_H_
#define LAUNCHPARAMS_H_

#include <cuda_runtime.h>
#include <glm/glm.hpp>
#include <owl/owl.h>
#include <owl/common/math/vec.h>

#include "Intrinsics.h"

struct LaunchParams
{
    /* Omnidirectional camera intrinsics. */
    Intrinsics intrinsics;

    /* Camera-to-world transforms. */
    glm::mat4 t_curr;
    glm::mat4 t_prev;

    /* Output buffers. */
    uint32_t    *fbDiffuse;
    uint16_t    *fbDepth;
    owl::vec4us *fbNormals;
    owl::vec4us *fbFlow;
    uint8_t     *fbOcclusion;   
    uint8_t     *coverage;

    uint8_t     renderFlags;

    /* Optix ray traversable object. */
    OptixTraversableHandle traversable;
};

struct TriangleMeshSBTData
{
    owl::vec3f *color;
    owl::vec3f *vertex;
    owl::vec3f *normal;
    owl::vec3f *texcoord;
    owl::vec3i *index;
    int hasTexture;
    cudaTextureObject_t texture;
};

#endif /* LAUNCHPARAMS_H_ */