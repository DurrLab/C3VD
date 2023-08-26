/***********************************************************************************/
/*
 *	File name:	Render.cu
 *
 *	Author:     Taylor Bobrow, Johns Hopkins University (2023)
 * 
 */

#include <cuda_runtime.h>
#include <glm/glm.hpp>
#include <optix_device.h>
#include <owl/common/math/random.h>

#include "Float3.h"
#include "LaunchParams.h"
#include "Quartic.cuh"
#include "RenderContext.h"
#include "RenderFlags.h"

#define MAX_DEPTH   100.0f
#define MAX_FLOW    20.0f

extern "C" __constant__ LaunchParams optixLaunchParams;

typedef owl::RayT<0, 2>  PrimaryRay;
typedef owl::RayT<1, 2>  OcclusionRay;

struct PrimaryRayPayload
{
    glm::vec3   hitPos;  /* Surface intersection point in world coord. */
    int         primID;  /* Primitive id of the intersected face. */
    float       diffuse; /* Diffuse lambertian shading. */
    float       depth;   /* Z-depth of intersection point. */
    glm::vec3   normal;  /* Surface normal of intersection point in camera space. */
};

struct OcclusionRayPayload
{
    glm::vec3   hitPos;  /* Surface intersection point in world coord. */
    int         primID;  /* Primitive id of the intersected face. */
};

/*  Forward project 3d vertex onto sensor plane using the 
    omnidirectional camera model. */
inline __device__
glm::vec2 forwardProjectVertex(const glm::vec3 v)
{
    float m = v.z / (pow(pow(v.x, 2.0f) + pow(v.y, 2.0f), 0.5f) + 1.0e-20f);

    double rho = solveQuartic((double)optixLaunchParams.intrinsics.polyCoeff.w,
                              (double)optixLaunchParams.intrinsics.polyCoeff.z,
                              (double)optixLaunchParams.intrinsics.polyCoeff.y,
                             -(double)m,
                              (double)optixLaunchParams.intrinsics.polyCoeff.x);

    glm::vec2 raw_uv;
    raw_uv.x = v.x / (pow(pow(v.x, 2.0f) + pow(v.y, 2.0f), 0.5f) + 1.0e-20f) * (float)rho;
    raw_uv.y = v.y / (pow(pow(v.x, 2.0f) + pow(v.y, 2.0f), 0.5f) + 1.0e-20f) * (float)rho;

    glm::vec2 ukr = optixLaunchParams.intrinsics.stretchMat*raw_uv + optixLaunchParams.intrinsics.center;

    return ukr;
}

/*  Generate a camera ray for pixel (ix,iy) in local 
    camera space using the omnidirectional model . */
inline __device__
glm::vec3 pixel2Ray(const glm::vec2 px)
{ 
    /* Convert to screen space. */
    glm::vec2 uvp = px - optixLaunchParams.intrinsics.center;

    /* Distort with stretch matrix. */
    glm::vec2 uvpp = glm::inverse(optixLaunchParams.intrinsics.stretchMat)*uvp;

    /*  Back-project pixel points onto the unit sphere by finding the coordinates
        of the vectors emanating from the single-effective-viewpoint to the unit
        sphere so that X^2 + Y^2 + Z^2 = 1. */
    float rho = sqrt( pow(uvpp.x,2) + pow(uvpp.y,2) );
    float z = optixLaunchParams.intrinsics.polyCoeff.x + 
             (0*rho) + 
             (optixLaunchParams.intrinsics.polyCoeff.y * pow(rho,2)) + 
             (optixLaunchParams.intrinsics.polyCoeff.z * pow(rho,3)) + 
             (optixLaunchParams.intrinsics.polyCoeff.w * pow(rho,4));

    /* Combine ray components and normalize. */
    glm::vec3 ray = glm::normalize(glm::vec3(uvpp.x,uvpp.y,z));

    return ray;
}

/*  Ray intersection program for primary rays - used
    to acquire basic geometric values. */
OPTIX_CLOSEST_HIT_PROGRAM(primary)()
{
    const TriangleMeshSBTData &sbtData = *(const TriangleMeshSBTData*)optixGetSbtDataPointer();

    PrimaryRayPayload &prd = owl::getPRD<PrimaryRayPayload>();

    const int primID = optixGetPrimitiveIndex();

    const owl::vec3i index = sbtData.index[primID];

    /* Basic hit information. */
    const glm::vec3 dir    = glm::make_vec3(&optixGetWorldRayDirection().x);
    const glm::vec3 origin = glm::make_vec3(&optixGetWorldRayOrigin().x);
    const float hit_t      = optixGetRayTmax();
    const glm::vec3 hitPos = origin + (hit_t*dir);

    /* Z-depth. */
    float depth = glm::vec3(glm::vec4(hitPos-origin,1.0)*optixLaunchParams.t_curr).z;

    /* Normal. */
    const owl::vec3f &A = sbtData.vertex[index.x];
    const owl::vec3f &B = sbtData.vertex[index.y];
    const owl::vec3f &C = sbtData.vertex[index.z];

    owl::vec3f N = normalize(owl::common::cross(B-A,C-A));

    glm::vec3 Nw = glm::make_vec3(&normalize(optixTransformNormalFromObjectToWorldSpace(N)).x);

    /* Face forward. */
    if (dot(dir,Nw) > 0.0) Nw = -Nw;

    /* Transform normal to screen space. */
    glm::vec3 Ns = glm::normalize(glm::mat3(glm::inverse(optixLaunchParams.t_curr))*Nw);

    /* Diffuse shading. */
    float diffuse = 0.1f + 0.9f*abs(dot(-1.f*Nw,dir));

    /* Store in ray payload. */
    prd.hitPos  = hitPos;
    prd.primID  = primID;
    prd.diffuse = diffuse;
    prd.depth   = depth;
    prd.normal  = Ns;
}

/*  Ray intersection program for occlusion rays - used
    to check for occluded faces. */
OPTIX_CLOSEST_HIT_PROGRAM(occlusion)()
{
    OcclusionRayPayload &prd = owl::getPRD<OcclusionRayPayload>();

    /* Basic hit information. */
    const int primID       = optixGetPrimitiveIndex();
    const glm::vec3 dir    = glm::vec3(optixGetWorldRayDirection().x,optixGetWorldRayDirection().y,optixGetWorldRayDirection().z);
    const glm::vec3 origin = glm::vec3(optixGetWorldRayOrigin().x,optixGetWorldRayOrigin().y,optixGetWorldRayOrigin().z);
    const float hit_t      = optixGetRayTmax();
    const glm::vec3 hitPos = origin + (hit_t*dir);

    /* Store in ray payload. */
    prd.hitPos = hitPos;
    prd.primID = primID;
}

/*  Ray generation program. */
OPTIX_RAYGEN_PROGRAM(render)()
{
    /* Current pixel index. */
    const glm::vec2 px(optixGetLaunchIndex().x,optixGetLaunchIndex().y);

    /* Initialize output values. */
    float       diffuse     = 0.f;
    float       depth       = 0.f;
    glm::vec3   normals     = glm::vec3(-1);
    glm::vec2   flow        = glm::vec2(-20);
    bool        occlusion   = 0;

    /* Generate ray in camera space. */
    glm::vec3 rayDirLocal = pixel2Ray(px);

    /* Transform ray to world space. */
    glm::vec3 rayDirWorld = glm::normalize(glm::vec3(optixLaunchParams.t_curr*
                                           glm::vec4(rayDirLocal,0.0f)));

    glm::vec3 rayOrigin = glm::vec3(optixLaunchParams.t_curr[3]);

    PrimaryRay ray(make_float3(rayOrigin),make_float3(rayDirWorld),1e-5f,1e4f);

    /* Initialize primary ray payload. */
    PrimaryRayPayload prd;
    prd.hitPos  = glm::vec3(0.0f);
    prd.primID  = -1;
    prd.diffuse = 0.0f;
    prd.depth   = 1.0f;
    prd.normal  = glm::vec3(0.0f);

    /* Trace ray with primary ray program and store results in PRD. */
    owl::traceRay(optixLaunchParams.traversable, ray, prd, OPTIX_RAY_FLAG_DISABLE_ANYHIT);

    /* If the ray intersected the model. */
    if(prd.primID != -1)
    {
        /* Diffuse shading. */
        diffuse = prd.diffuse;

        /* Depth. */
        if(optixLaunchParams.renderFlags & RenderFlags::DEPTH)
            depth = prd.depth;
        
        /* Screen space normals. */
        if(optixLaunchParams.renderFlags & RenderFlags::SCREEN_SPACE_NORMALS)
            normals = prd.normal;

        /* Optical flow. */
        if(optixLaunchParams.renderFlags & RenderFlags::OPTICAL_FLOW)
        {
            /* Transform intersection point from world space to previous frame camera coordinate system. */
            glm::vec3 v_cam_prev = glm::vec3(glm::inverse(optixLaunchParams.t_prev)*glm::vec4(prd.hitPos,1.0));
            
            /* Forward project the point to the sensor plane. */
            glm::vec2 px_prev = forwardProjectVertex(v_cam_prev);

            /* If the pixel is within the frame. */
            if(px_prev.x < optixLaunchParams.intrinsics.size.x &&
               px_prev.x >= 0 &&
               px_prev.y < optixLaunchParams.intrinsics.size.y &&
               px_prev.y >= 0)
            {
                /*  Check whether a ray cast from the previous frame location intersects a face
                    with a different primitive id, indicating that it was occluded. */
                glm::vec3 rayDirLocal_prev = pixel2Ray(px_prev);

                glm::vec3 rayDirWorld_prev = glm::normalize(glm::vec3(optixLaunchParams.t_prev*glm::vec4(rayDirLocal_prev,0.0f)));

                glm::vec3 rayOrigin_prev = glm::vec3(optixLaunchParams.t_prev[3]);

                OcclusionRay ray_prev(make_float3(rayOrigin_prev),make_float3(rayDirWorld_prev),1e-5f,1e4f);

                OcclusionRayPayload prd_prev;
                prd_prev.hitPos = glm::vec3(0.0);
                prd_prev.primID = -1;

                owl::traceRay(optixLaunchParams.traversable, ray_prev, prd_prev, OPTIX_RAY_FLAG_DISABLE_ANYHIT);

                /*  If the intersection primitive id matches, then 
                    it is not occluded -> save the flow. */
                if(prd.primID == prd_prev.primID)
                    flow = px_prev - px;        
            }
        }
        
        /* Occlusion. */
        if(optixLaunchParams.renderFlags & RenderFlags::OCCLUSION)
        {
            OcclusionRay ray_occ(make_float3(prd.hitPos + rayDirWorld*1.0e-4f),make_float3(rayDirWorld),1e-5f,1e4f);

            OcclusionRayPayload prd_occ;
            prd_occ.hitPos = glm::vec3(0.0);
            prd_occ.primID = -1;

            owl::traceRay(optixLaunchParams.traversable, ray_occ, prd_occ, OPTIX_RAY_FLAG_DISABLE_ANYHIT);

            /* If intersected and within the max depth. */
            if(prd_occ.primID!=-1 && (glm::vec3(optixLaunchParams.t_curr*glm::vec4(prd_occ.hitPos,1.0)).z <= 100.0))
                occlusion = 1;
        }

        /* Coverage. */
        if(optixLaunchParams.renderFlags & RenderFlags::COVERAGE)
        {
            /* Mark face in coverage texture if intersected. */
            if(prd.primID != -1)
                optixLaunchParams.coverage[(uint32_t)prd.primID] = (uint8_t)255;
        }
    }

    /* Store values in frame buffers. */
    const uint32_t px_idx = px.x + px.y*optixLaunchParams.intrinsics.size.x;

    optixLaunchParams.fbDiffuse[px_idx] = owl::make_rgba(owl::vec3f(diffuse));

    optixLaunchParams.fbDepth[px_idx] = (uint16_t)65535.0*(owl::clamp(depth / MAX_DEPTH,0.f,1.f));

    optixLaunchParams.fbNormals[px_idx] = owl::vec4us((uint16_t)(65535.0*((normals.x + 1.f)/2.f)),
                                                      (uint16_t)(65535.0*((normals.y + 1.f)/2.f)),
                                                      (uint16_t)(65535.0*((normals.z + 1.f)/2.f)),
                                                       65535);

    optixLaunchParams.fbFlow[px_idx] = owl::vec4us((uint16_t)(65535.0*((flow.x + MAX_FLOW)/(2.f*MAX_FLOW))),
                                                   (uint16_t)(65535.0*((flow.y + MAX_FLOW)/(2.f*MAX_FLOW))),
                                                    0,
                                                    65535);

    optixLaunchParams.fbOcclusion[px_idx] = occlusion > 0 ? (uint8_t)255 : (uint8_t)0;
}
