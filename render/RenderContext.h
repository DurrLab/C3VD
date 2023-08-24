/***********************************************************************************/
/*
 *	File name:	RenderContext.h
 *
 *	Author:     Taylor Bobrow, Johns Hopkins University (2023)
 * 
 */

#ifndef RENDERCONTEXT_H_
#define RENDERCONTEXT_H_

#include <glm/glm.hpp>
#include <glm/gtc/matrix_access.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <owl/owl.h>
#include <owl/owl_host.h>

#include "Intrinsics.h"
#include "LaunchParams.h"
#include "Model.h"
#include "RenderFlags.h"
#include "TransformFlags.h"

class RenderContext
{
    public:
        RenderContext(const Model *model, const Intrinsics &intrinsics);

        void updateCameraTransform(glm::mat4 tform, bool flag);

        void updateMeshTransform(glm::mat4 tform);

        void updateRenderFlags(uint8_t flags);

        void render(void);

    public:
        /* Output frame buffers. */
        OWLBuffer fbDiffuse   = nullptr;
        OWLBuffer fbDepth     = nullptr;
        OWLBuffer fbNormals   = nullptr;
        OWLBuffer fbFlow      = nullptr;
        OWLBuffer fbOcclusion = nullptr;
        OWLBuffer coverage    = nullptr;

    private:
        void buildAccel(void);

    private:
        const Intrinsics    intrinsics;
        const Model         *model;

        /* Optix resources. */
        OWLContext      context         = nullptr;
        OWLModule       module          = nullptr;
        OWLLaunchParams launchParams    = nullptr;
        OWLRayGen       rayGen          = nullptr;
        OWLGroup        world           = nullptr;

};
#endif /* RENDERCONTEXT_H_ */