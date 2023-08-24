/***********************************************************************************/
/*
 *	File name:	RenderContext.cpp
 *
 *	Author:     Taylor Bobrow, Johns Hopkins University (2023)
 *
 *  To-do:      Update the coverage implementation to track for multiple meshes -
 *              instead of only mesh[0].
 */

#include "RenderContext.h"

extern "C" char ptxCode[];

RenderContext::RenderContext(const Model *model, const Intrinsics &intrinsics)
:   intrinsics(intrinsics),
    model(model)
{
    /* Create an optix context. */
    context = owlContextCreate(nullptr, 1);

    /* Set the number of ray types to 2:
        1.) Primary ray
        2.) Occlusion ray
    */
    owlContextSetRayTypeCount(context, 2);

    /* Create an optix module. */
    module = owlModuleCreate(context,ptxCode);

    /* Designate the ray generation program as "render". */
    rayGen = owlRayGenCreate(context,module,"render",0,nullptr,-1);

    /* Set up launch parameters accessible in render program. */
    OWLVarDecl launchParamsVars[] =
    {
        { "intrinsics",     OWL_USER_TYPE(Intrinsics),  OWL_OFFSETOF(LaunchParams,intrinsics)},
        { "t_curr",         OWL_USER_TYPE(glm::mat4),   OWL_OFFSETOF(LaunchParams,t_curr)},
        { "t_prev",         OWL_USER_TYPE(glm::mat4),   OWL_OFFSETOF(LaunchParams,t_prev)},
        { "fbDiffuse",      OWL_BUFPTR,                 OWL_OFFSETOF(LaunchParams,fbDiffuse)},
        { "fbDepth",        OWL_BUFPTR,                 OWL_OFFSETOF(LaunchParams,fbDepth)},
        { "fbNormals",      OWL_BUFPTR,                 OWL_OFFSETOF(LaunchParams,fbNormals)},
        { "fbFlow",         OWL_BUFPTR,                 OWL_OFFSETOF(LaunchParams,fbFlow)},
        { "fbOcclusion",    OWL_BUFPTR,                 OWL_OFFSETOF(LaunchParams,fbOcclusion)},
        { "coverage",       OWL_BUFPTR,                 OWL_OFFSETOF(LaunchParams,coverage)},
        { "renderFlags",    OWL_USER_TYPE(uint8_t),     OWL_OFFSETOF(LaunchParams,renderFlags)},
        { "world",          OWL_GROUP,                  OWL_OFFSETOF(LaunchParams,traversable)},
        { }
    };
    launchParams = owlParamsCreate(context,sizeof(LaunchParams),launchParamsVars,-1);
    
    /* Build the acceleration structure from the 3D model. */
    buildAccel();

    /* Setup Optix rendering pipeline. */
    owlBuildPrograms(context);
    
    owlBuildPipeline(context);
    
    owlBuildSBT(context);

    /* Intialize launch parameters. */
    owlParamsSetRaw(launchParams,"intrinsics", &intrinsics);

    glm::mat4 identity( 1.0f );
    owlParamsSetRaw(launchParams,"t_curr", &identity);
    owlParamsSetRaw(launchParams,"t_prev", &identity);
 
    uint8_t flags = RenderFlags::DIFFUSE;
    owlParamsSetRaw(launchParams,"renderFlags", &flags);

    /* Create a buffer in device memory for each frame and initialize to 0 . */
    fbDiffuse   = owlDeviceBufferCreate(context,OWL_UINT4,intrinsics.size.x*intrinsics.size.y,nullptr);
    fbDepth     = owlDeviceBufferCreate(context,OWL_UINT2,intrinsics.size.x*intrinsics.size.y,nullptr);
    fbNormals   = owlDeviceBufferCreate(context,OWL_USHORT4,intrinsics.size.x*intrinsics.size.y,nullptr);
    fbFlow      = owlDeviceBufferCreate(context,OWL_USHORT4,intrinsics.size.x*intrinsics.size.y,nullptr);
    fbOcclusion = owlDeviceBufferCreate(context,OWL_UINT,intrinsics.size.x*intrinsics.size.y,nullptr);
    owlBufferClear(fbDiffuse);
    owlBufferClear(fbDepth);
    owlBufferClear(fbNormals);
    owlBufferClear(fbFlow);
    owlBufferClear(fbOcclusion);

    /*  Create a buffer in device memory with one entry per face 
        in the 3D model and initialize all to *FALSE*. Entries will
        be progressively toggled to *TRUE* as model faces are intersected
        in the render program
    */
    coverage    = owlDeviceBufferCreate(context,OWL_UINT,model->meshes[0]->index.size(),nullptr);
    owlBufferClear(coverage);

    /* Make buffers accessible through launch params. */
    owlParamsSetBuffer(launchParams,"fbDiffuse",    fbDiffuse);
    owlParamsSetBuffer(launchParams,"fbDepth",      fbDepth);
    owlParamsSetBuffer(launchParams,"fbNormals",    fbNormals);
    owlParamsSetBuffer(launchParams,"fbFlow",       fbFlow);
    owlParamsSetBuffer(launchParams,"fbOcclusion",  fbOcclusion);
    owlParamsSetBuffer(launchParams,"coverage",     coverage);
}


/*  Updates the camera to world transform to the input homogenous transform.
    Use *CURRENT_TRANSFORM* to update the current camera pose, and use 
    use *PREVIOUS_TRANSFORM* to update the previous frame camera pose (for
    rendering optical flow).
*/
void RenderContext::updateCameraTransform(glm::mat4 tform, bool flag)
{
    if(flag==TransformFlags::CURRENT_TRANSFORM)
        owlParamsSetRaw(launchParams,"t_curr",&tform);
    else if(flag==TransformFlags::PREVIOUS_TRANSFORM)
        owlParamsSetRaw(launchParams,"t_prev",&tform);
    else
    {
        printf("\x1B[31mA second argument to updateCameraTransform not a valid TransformFlag\n\x1B[0m");
        exit(EXIT_FAILURE);
    }
}

/*  Updates the model transform to the input homogenous transform. */
void RenderContext::updateMeshTransform(const glm::mat4 tform)
{
    /* OWL expects the upper 3x4 part of the homogenous matrix, in column major order*/
    float colMajorTransform[12];
    for (int c = 0; c < 4; ++c)
    {
        colMajorTransform[c*3 + 0] = glm::column(tform,c)[0];
        colMajorTransform[c*3 + 1] = glm::column(tform,c)[1];
        colMajorTransform[c*3 + 2] = glm::column(tform,c)[2];
    }

    /* Update the transform in the accel structure. */
    owlInstanceGroupSetTransform(world, 0, colMajorTransform, OWL_MATRIX_FORMAT_OWL);

    /* Refit the accel structure with the new transform. */
    owlGroupBuildAccel(world);

    /* Update the SBT data. */
    owlBuildSBT(context);
}

/* Updates which outputs to render. */
void RenderContext::updateRenderFlags(uint8_t flags)
{
    owlParamsSetRaw(launchParams,"renderFlags", &flags);
}

/*  Render a frame using the current launch parameters
    and model transform. 
*/
void RenderContext::render(void)
{
    owlLaunch2D(rayGen, intrinsics.size.x, intrinsics.size.y, launchParams);

    cudaDeviceSynchronize();
}

/* Build an OWL acceleration structure from meshes. */
void RenderContext::buildAccel(void)
{
    const int numMeshes = (int)model->meshes.size();

    std::vector<OWLGeom> meshes;

    OWLVarDecl triMeshVars[] = {
        { "color",      OWL_FLOAT3, OWL_OFFSETOF(TriangleMeshSBTData,color) },
        { "vertex",     OWL_BUFPTR, OWL_OFFSETOF(TriangleMeshSBTData,vertex) },
        { "normal",     OWL_BUFPTR, OWL_OFFSETOF(TriangleMeshSBTData,normal) },
        { "index",      OWL_BUFPTR, OWL_OFFSETOF(TriangleMeshSBTData,index) },
        { "texcoord",   OWL_BUFPTR, OWL_OFFSETOF(TriangleMeshSBTData,texcoord) },
        { "hasTexture", OWL_INT,    OWL_OFFSETOF(TriangleMeshSBTData,hasTexture) },
        { "texture",    OWL_TEXTURE,OWL_OFFSETOF(TriangleMeshSBTData,texture) },
        { nullptr }
    };
    OWLGeomType triMeshGeomType = owlGeomTypeCreate(context,OWL_GEOM_TRIANGLES,sizeof(TriangleMeshSBTData),triMeshVars,-1);

    /* Designate the closest hit programs for each ray type. */
    owlGeomTypeSetClosestHit(triMeshGeomType,0,module,"primary");
    owlGeomTypeSetClosestHit(triMeshGeomType,1,module,"occlusion");

    /* Create textures. */
    std::vector<OWLTexture> textures;

    int numTextures = (int)model->textures.size();

    textures.resize(numTextures);

    for (int textureID=0;textureID<numTextures;textureID++)
    {
        auto texture = model->textures[textureID];
        textures[textureID] = owlTexture2DCreate(context,OWL_TEXEL_FORMAT_RGBA8,
                                                 texture->resolution.x,texture->resolution.y,
                                                 texture->pixel,OWL_TEXTURE_LINEAR,OWL_TEXTURE_CLAMP);
    }

    /* Create the model geometry. */
    std::vector<OWLGeom> geoms;
    for (int meshID=0;meshID<numMeshes;meshID++)
    {
        TriangleMesh &mesh = *model->meshes[meshID];

        OWLBuffer vertexBuffer      = owlDeviceBufferCreate(context,OWL_FLOAT3,mesh.vertex.size(),mesh.vertex.data());
        OWLBuffer indexBuffer       = owlDeviceBufferCreate(context,OWL_INT3,mesh.index.size(),mesh.index.data());
        OWLBuffer normalBuffer      = mesh.normal.empty() ? nullptr : 
                                    owlDeviceBufferCreate(context,OWL_FLOAT3,mesh.normal.size(),mesh.normal.data());
        OWLBuffer texcoordBuffer    = mesh.texcoord.empty() ? nullptr : 
                                    owlDeviceBufferCreate(context,OWL_FLOAT2,mesh.texcoord.size(),mesh.texcoord.data());
    
        OWLGeom geom = owlGeomCreate(context,triMeshGeomType);

        owlTrianglesSetVertices(geom,vertexBuffer,mesh.vertex.size(),sizeof(owl::vec3f),0);
        owlTrianglesSetIndices(geom,indexBuffer,mesh.index.size(),sizeof(owl::vec3i),0);

        owlGeomSetBuffer(geom,"index",indexBuffer);
        owlGeomSetBuffer(geom,"vertex",vertexBuffer);
        owlGeomSetBuffer(geom,"normal",normalBuffer);
        owlGeomSetBuffer(geom,"texcoord",texcoordBuffer);

        owlGeomSet3f(geom,"color",(const owl3f &)mesh.diffuse);
        if (mesh.diffuseTextureID >= 0) 
        {
            owlGeomSet1i(geom,"hasTexture",1);
            assert(mesh.diffuseTextureID < (int)textures.size());
            owlGeomSetTexture(geom,"texture",textures[mesh.diffuseTextureID]);
        }
        else
        {
            owlGeomSet1i(geom,"hasTexture",0);
        }
        geoms.push_back(geom);
    }
    /* Combine into a triangle group. */
    OWLGroup triGroup = owlTrianglesGeomGroupCreate(context,geoms.size(),geoms.data());
    owlGroupBuildAccel(triGroup);

    /* Build the acceleration structure. */
    world = owlInstanceGroupCreate(context,1);
    owlInstanceGroupSetChild(world,0,triGroup);
    owlGroupBuildAccel(world);
    owlParamsSetGroup(launchParams,"world",world);
    owlGroupBuildAccel(world);
}