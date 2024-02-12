/***********************************************************************************/
/*
 *	File name:	RenderingModule.cpp
 *
 *	Author:     Taylor Bobrow, Johns Hopkins University (2023)
 * 
 */

#include "RenderingModule.h"

#include <stdexcept>
#include <glm/gtx/string_cast.hpp>
#include <glm/gtc/matrix_access.hpp>

#include "render/Intrinsics.h"
#include "render/RenderFlags.h"
#include "render/Rgba2rgb.cuh"
#include "render/TransformFlags.h"
#include "tools/ConfigParser.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

RenderingModule::RenderingModule(int argc, char* argv[])
{
    /* Load parameters from configuration file. */
    if (argc < 3)
    {
        printf("\x1B[31mA second argument to the configuration file path must be included\n\x1B[0m");
        exit(EXIT_FAILURE);
    }
    loadParams(argv[2]);

    modelFilePath      = std::string(argv[2]) + "model.obj";
    poseFilePath       = std::string(argv[2]) + "pose.txt";
    rgbFolderPath      = std::string(argv[2]) + "rgb/";
    maskFilePath        = std::string(argv[2]) + "mask.png";
    renderFolderPath   = std::string(argv[2]) + "render/";

    /* Load obj file. */
    model = loadOBJ(modelFilePath);

    /* Create intrinsics object. */
    Intrinsics intrinsics(width, height, cx, cy, a0, a1, a2, a3, a4, c, d, e);

    /* Handeye object. */
    handeye = new Handeye(A_cal, B_cal, X);

    /* Mask. */
    mask = new Mask(pangolin::LoadImage(maskFilePath));

    /* Pose log. */
    poseLog = new PoseLog(poseFilePath);

    /* Inquire the number of frames in the target folder. */
    numFrames = getFrameCount(rgbFolderPath);

    /* Create optix rendering context. */
    context = new RenderContext(model, intrinsics);

    /* Enable all render flags. */
    context->updateRenderFlags(RenderFlags::DEPTH
                             | RenderFlags::SCREEN_SPACE_NORMALS
                             | RenderFlags::OPTICAL_FLOW
                             | RenderFlags::OCCLUSION
                             | RenderFlags::COVERAGE);

    /* Set the model transform to the ground truth. */
    glm::quat qx = glm::angleAxis(modelTransformR6[0],glm::vec3(1.0,0.0,0.0));
    glm::quat qy = glm::angleAxis(modelTransformR6[1],glm::vec3(0.0,1.0,0.0));
    glm::quat qz = glm::angleAxis(modelTransformR6[2],glm::vec3(0.0,0.0,1.0));
    glm::quat q  = qz*qy*qx;

    T_final = glm::mat4_cast(q);
    T_final = glm::column(T_final,3,glm::vec4(modelTransformR6[3],modelTransformR6[4],modelTransformR6[5],1.0f));

    context->updateMeshTransform(T_final);

    /* Allocate device memory for image data w/ alpha channel removed. */
    cudaMalloc((void**) &normalsNoAlpha_dev, sizeof(uint16_t)*width*height*3);
    cudaMalloc((void**) &flowNoAlpha_dev,    sizeof(uint16_t)*width*height*3);

    /* Allocate host memory for rendered image data. */
    depth_host      = (uint16_t*)malloc(width*height*1*sizeof(uint16_t));
    normals_host    = (uint16_t*)malloc(width*height*4*sizeof(uint16_t));
    flow_host       = (uint16_t*)malloc(width*height*4*sizeof(uint16_t));
    occlusion_host  = (uint8_t*) malloc(width*height*1*sizeof(uint8_t));
    coverage_host   = (uint8_t*) malloc(model->meshes[0]->index.size()*sizeof(uint8_t));

    /* Progress bar. */
    bar = new progressbar(numFrames);
    bar->set_opening_bracket_char("Rendering progress: {");
}

RenderingModule::~RenderingModule(void)
{
    delete bar;
    delete handeye;
    delete mask;
    delete model;
    delete poseLog;
    delete context;
    cudaFree(normalsNoAlpha_dev);
    cudaFree(flowNoAlpha_dev);
    free(depth_host);
    free(normals_host);
    free(flow_host);
    free(occlusion_host);
}

void RenderingModule::launch(void)
{
    std::ofstream poseFile;
    poseFile.open(renderFolderPath + "pose.txt",std::ios_base::app);

    /* Iterate through each frame and render. */
    for(int n = 0; n < numFrames; n++)
    {
        /* Update camera transform (current). */
        glm::mat4 A1_curr = poseLog->getTransform(n * (1.0f / FPS) + poseStartTime);
        glm::mat4 B1_curr = handeye->A2B(A1_curr);
        context->updateCameraTransform(B1_curr,TransformFlags::CURRENT_TRANSFORM);

        /* Update camera transform (previous). */
        if(n>0)
        {
            glm::mat4 A1_prev = poseLog->getTransform((n-1) * (1.0f / FPS) + poseStartTime);
            glm::mat4 B1_prev = handeye->A2B(A1_prev);
            context->updateCameraTransform(B1_prev,TransformFlags::PREVIOUS_TRANSFORM);
        }

        /* Render frame. */
        context->render();

        /* Mask corners. */
        mask->apply((uint32_t*)owlBufferGetPointer(context->fbDiffuse,0));
        mask->apply((uint16_t*)owlBufferGetPointer(context->fbDepth,0));
        mask->apply((uint64_t*)owlBufferGetPointer(context->fbNormals,0));
        mask->apply((uint64_t*)owlBufferGetPointer(context->fbFlow,0));
        mask->apply((uint8_t*) owlBufferGetPointer(context->fbOcclusion,0));

        /* Remove alpha channel from normals/flow. */
        rgba2Rgb((uint16_t*)owlBufferGetPointer(context->fbNormals,0),normalsNoAlpha_dev,width,height);
        rgba2Rgb((uint16_t*)owlBufferGetPointer(context->fbFlow,0),flowNoAlpha_dev,width,height);

        /* Copy rendering data from device to host. */
        cudaMemcpy(depth_host,owlBufferGetPointer(context->fbDepth,0),width*height*1*sizeof(uint16_t),cudaMemcpyDeviceToHost);
        cudaMemcpy(normals_host,normalsNoAlpha_dev,width*height*3*sizeof(uint16_t),cudaMemcpyDeviceToHost);
        cudaMemcpy(flow_host,flowNoAlpha_dev,width*height*3*sizeof(uint16_t),cudaMemcpyDeviceToHost);
        cudaMemcpy(occlusion_host,owlBufferGetPointer(context->fbOcclusion,0),width*height*1*sizeof(uint8_t),cudaMemcpyDeviceToHost);

        /* Save frames. */
        std::string num_str = std::string(4 - std::min(4, (int)std::to_string(n).length()), '0') + std::to_string(n);

        std::string depthFilename = renderFolderPath + num_str + "_depth.tiff";
        TinyTIFFWriterFile* depthTiff = TinyTIFFWriter_open(depthFilename.c_str(),16,TinyTIFFWriter_UInt,1,width,height,TinyTIFFWriter_Greyscale);
        TinyTIFFWriter_writeImage(depthTiff, depth_host);
        TinyTIFFWriter_close(depthTiff);

        std::string normalsFilename = renderFolderPath + num_str + "_normals.tiff";
        TinyTIFFWriterFile* normalsTiff = TinyTIFFWriter_open(normalsFilename.c_str(),16,TinyTIFFWriter_UInt,3,width,height,TinyTIFFWriter_RGB);
        TinyTIFFWriter_writeImage(normalsTiff, normals_host);
        TinyTIFFWriter_close(normalsTiff);

        if(n>0)
        {
            std::string flowFilename = renderFolderPath + num_str + "_flow.tiff";
            TinyTIFFWriterFile* flowTiff = TinyTIFFWriter_open(flowFilename.c_str(),16,TinyTIFFWriter_UInt,3,width,height,TinyTIFFWriter_RGB);
            TinyTIFFWriter_writeImage(flowTiff, flow_host);
            TinyTIFFWriter_close(flowTiff);
        }

        std::string occlusionFilename = renderFolderPath + num_str + "_occlusion.png";
        stbi_write_png(occlusionFilename.c_str(),width,height,1,occlusion_host,width*sizeof(uint8_t));

        /* Save pose. */
        poseFile << B1_curr[0][0] << "," << B1_curr[0][1] << "," << B1_curr[0][2] << "," << B1_curr[0][3] << ","
                 << B1_curr[1][0] << "," << B1_curr[1][1] << "," << B1_curr[1][2] << "," << B1_curr[1][3] << ","
                 << B1_curr[2][0] << "," << B1_curr[2][1] << "," << B1_curr[2][2] << "," << B1_curr[2][3] << ","
                 << B1_curr[3][0] << "," << B1_curr[3][1] << "," << B1_curr[3][2] << "," << B1_curr[3][3] << "\n";
    
        bar->update();
    }

    poseFile.close();

    /* Save model with coverage texture. */
    cudaMemcpy(coverage_host, owlBufferGetPointer(context->coverage,0), model->meshes[0]->index.size(), cudaMemcpyDeviceToHost);

    writeOBJ(renderFolderPath + "coverage_mesh.obj", model, coverage_host, T_final);

    std::cout << "\033[1;32m\nRendering complete.\033[0m\n";
}

void RenderingModule::loadParams(std::string filepath)
{
    ConfigParser parser = ConfigParser(filepath + "config.ini");

    width   = (unsigned int)parser.aConfig<int>("width");
    height  = (unsigned int)parser.aConfig<int>("height");
    cx      = parser.aConfig<float>("cx");
    cy      = parser.aConfig<float>("cy");
    a0      = parser.aConfig<float>("a0");
    a1      = parser.aConfig<float>("a1");
    a2      = parser.aConfig<float>("a2");
    a3      = parser.aConfig<float>("a3");
    a4      = parser.aConfig<float>("a4");
    c       = parser.aConfig<float>("c");
    d       = parser.aConfig<float>("d");
    e       = parser.aConfig<float>("e");

    A_cal   = glm::make_mat4(parser.aConfigVec<float>("A_cal").data());
    B_cal   = glm::make_mat4(parser.aConfigVec<float>("B_cal").data());
    X       = glm::make_mat4(parser.aConfigVec<float>("X").data());

    modelTransformR6 = parser.aConfigVec<float>("modelTransform");
    poseStartTime = parser.aConfig<float>("poseStartTime");
}

unsigned int RenderingModule::getFrameCount(std::string directoryPath)
{
    DIR *direc;
    struct dirent *entry;
    direc = opendir(directoryPath.c_str());
    unsigned int count = 0;
    if (direc)
    {
        while ((entry = readdir(direc)) != NULL)
        {
            if(strstr(entry->d_name,".png"))
                count++;
        }
        closedir(direc); //close all directory
        printf("Identified %d .png frames in folder %s\n", count, directoryPath.c_str());
    }
    return count;
}

void RenderingModule::writeOBJ(const std::string &filename,const Model *model,const uint8_t *coverageTex, glm::mat4 modelTransform)
{
    /* Just handle first mesh*/
    const TriangleMesh *mesh = model->meshes[0];

    std::ofstream modelFile;
    modelFile.open(filename.c_str());

    /* Output vertices. */
    for (int i = 0; i < mesh->vertex.size(); i++)
    {   
        auto v = mesh->vertex[i];
        
        glm::vec3 v_tform = glm::vec3(T_final * glm::vec4(v.x,v.y,v.z,1.0));

        modelFile << "v "
                  << v_tform.x << " "
                  << v_tform.y << " "
                  << v_tform.z << "\n";
    }

    /* Output vertex texture coords for coverage
            [0 0]: observed
            [1 1]: missed */
    modelFile << "vt 0.0 0.0\n"
              << "vt 1.0 1.0\n";

    /* Output faces and coverage texture coords. */
    int count = 0;
    for (int i = 0; i < mesh->index.size(); i++)
    {
        /* If observed, use text coord 0, otherwise 1. */
        int vt = (int)coverageTex[i] == 255 ? 1 : 2;
        modelFile << "f "
                  << mesh->index[i].x + 1 << "/" << vt << " "
                  << mesh->index[i].y + 1 << "/" << vt << " "
                  << mesh->index[i].z + 1 << "/" << vt << "\n";
    }

    modelFile.close();
}