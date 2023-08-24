/***********************************************************************************/
/*
 *	File name:	AlignmentModule.cpp
 *
 *	Author:     Taylor Bobrow, Johns Hopkins University (2023)
 * 
 */

#include "AlignmentModule.h"

#include <stdexcept>
#include <glm/gtx/string_cast.hpp>
#include <glm/gtc/matrix_access.hpp>

#include "render/Intrinsics.h"
#include "render/RenderFlags.h"
#include "render/TransformFlags.h"
#include "tools/ConfigParser.h"

AlignmentModule::AlignmentModule(int argc, char* argv[])
{
    /* Load parameters from configuration file. */
    if (argc < 3)
    {
        printf("\x1B[31mA second argument with the configuration file path must be included\n\x1B[0m");
        exit(EXIT_FAILURE);
    }
    loadParams(argv[2]);

    modelFilePath   = std::string(argv[2]) + "model.obj";
    poseFilePath    = std::string(argv[2]) + "pose.txt";
    rgbFolderPath   = std::string(argv[2]) + "rgb/";
    maskFilePath    = std::string(argv[2]) + "mask.png";

    /* Load obj file. */
    model = loadOBJ(modelFilePath);

    /* Intrinsics object. */
    Intrinsics intrinsics(width, height, cx, cy, a0, a1, a2, a3, a4, c, d, e);

    /* Handeye object. */
    handeye = new Handeye(A_cal, B_cal, X);

    /* Mask object. */
    mask = new Mask(pangolin::LoadImage(maskFilePath));

    /* Pose log. */
    poseLog = new PoseLog(poseFilePath);

    /* Inquire the number of frames in the target folder. */
    numFrames = getFrameCount(rgbFolderPath);

    /* Create optix rendering context. */
    context = new RenderContext(model, intrinsics);

    /* Initialize GUI values. */
    gui = new Gui(width, height);
    gui->currentFrame->Meta().range[0] = 0;
    gui->currentFrame->Meta().range[1] = numFrames-1;
    gui->poseStartTime->Meta().range[0] = 0;
    gui->poseStartTime->Meta().range[1] = poseLog->getEndTime() - ((numFrames) * (1.0f / FPS));

    gui->poseStartTime->Ref().Set(poseStartTime);
    gui->modelT_pitch->Ref().Set(modelTransformR6[0]);
    gui->modelT_yaw->Ref().Set(modelTransformR6[1]);
    gui->modelT_roll->Ref().Set(modelTransformR6[2]);
    gui->modelT_x->Ref().Set(modelTransformR6[3]);
    gui->modelT_y->Ref().Set(modelTransformR6[4]);
    gui->modelT_z->Ref().Set(modelTransformR6[5]);

    gui->previewDepth->Ref().Set(true);
    gui->previewNormals->Ref().Set(true);
    gui->previewFlow->Ref().Set(true);
    gui->previewOcclusion->Ref().Set(true);

    /* Start with all render flags except coverage. */
    context->updateRenderFlags(RenderFlags::DEPTH
                             | RenderFlags::SCREEN_SPACE_NORMALS
                             | RenderFlags::OPTICAL_FLOW
                             | RenderFlags::OCCLUSION);
}

AlignmentModule::~AlignmentModule(void)
{
    delete gui;
    delete handeye;
    delete mask;
    delete model;
    delete poseLog;
    delete context;
}

void AlignmentModule::launch(void)
{
    /* Main update loop. */
    while(!pangolin::ShouldQuit())
    {
        /* Update model transform. */
        glm::quat qx = glm::angleAxis(gui->modelT_pitch->Get(),glm::vec3(1.0,0.0,0.0));
        glm::quat qy = glm::angleAxis(gui->modelT_yaw->Get(),glm::vec3(0.0,1.0,0.0));
        glm::quat qz = glm::angleAxis(gui->modelT_roll->Get(),glm::vec3(0.0,0.0,1.0));
        glm::quat q  = qz*qy*qx;

        glm::mat4 T_m = glm::mat4_cast(q);
        T_m = glm::column(T_m,3,glm::vec4(gui->modelT_x->Get(),gui->modelT_y->Get(),gui->modelT_z->Get(),1.0f));

        context->updateMeshTransform(T_m);

        /* uUpdate camera transform (current). */
        glm::mat4 A1_curr = poseLog->getTransform((gui->currentFrame->Get()) * (1.0f / FPS) + gui->poseStartTime->Get());
        glm::mat4 B1_curr = handeye->A2B(A1_curr);
        context->updateCameraTransform(B1_curr,TransformFlags::CURRENT_TRANSFORM);

        /* Update camera transform (previous). */
        if(gui->currentFrame->Get()>0)
        {
            glm::mat4 A1_prev = poseLog->getTransform((gui->currentFrame->Get()-1) * (1.0f / FPS) + gui->poseStartTime->Get());
            glm::mat4 B1_prev = handeye->A2B(A1_prev);
            context->updateCameraTransform(B1_prev,TransformFlags::PREVIOUS_TRANSFORM);
        }
        
        /* Update render flags. */
        uint8_t flags =  RenderFlags::DIFFUSE
                      | (RenderFlags::DEPTH                 & (uint8_t)255*gui->previewDepth->Get())
                      | (RenderFlags::SCREEN_SPACE_NORMALS  & (uint8_t)255*gui->previewNormals->Get())
                      | (RenderFlags::OPTICAL_FLOW          & (uint8_t)255*gui->previewFlow->Get())
                      | (RenderFlags::OCCLUSION             & (uint8_t)255*gui->previewOcclusion->Get());

        context->updateRenderFlags(flags);

        /* Render frame. */
        context->render();

        /* If overlaying target frames. */
        if(gui->overlayTarget->Get())
            gui->loadTargetImg(rgbFolderPath + std::to_string(gui->currentFrame->Get()) + ".png");

        /* Mask corners. */
        mask->apply((uint32_t*)owlBufferGetPointer(context->fbDiffuse,0));
        mask->apply((uint16_t*)owlBufferGetPointer(context->fbDepth,0));
        mask->apply((uint64_t*)owlBufferGetPointer(context->fbNormals,0));
        mask->apply((uint64_t*)owlBufferGetPointer(context->fbFlow,0));
        mask->apply((uint8_t*) owlBufferGetPointer(context->fbOcclusion,0));

        /* Copy to gl textures. */
        gui->copyToDisplayBuffer(owlBufferGetPointer(context->fbDiffuse,0),   RenderFlags::DIFFUSE);
        gui->copyToDisplayBuffer(owlBufferGetPointer(context->fbDepth,0),     RenderFlags::DEPTH);
        gui->copyToDisplayBuffer(owlBufferGetPointer(context->fbNormals,0),   RenderFlags::SCREEN_SPACE_NORMALS);
        gui->copyToDisplayBuffer(owlBufferGetPointer(context->fbFlow,0),      RenderFlags::OPTICAL_FLOW);
        gui->copyToDisplayBuffer(owlBufferGetPointer(context->fbOcclusion,0), RenderFlags::OCCLUSION);

        /* Display gui frame. */
        gui->render();
    }
}

void AlignmentModule::loadParams(std::string filepath)
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

unsigned int AlignmentModule::getFrameCount(std::string directoryPath)
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