/***********************************************************************************/
/*
 *	File name:	RenderingModule.h
 *
 *	Author:     Taylor Bobrow, Johns Hopkins University (2023)
 * 
 */

#ifndef RENDERINGMODULE_H_
#define RENDERINGMODULE_H_

#include <dirent.h>
#include <stdio.h>
#include <string>

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <owl/common/math/vec.h>
#include <pangolin/image/image_io.h>
#include <pangolin/pangolin.h>
#include <tinytiffwriter.h>

#include "render/LaunchParams.h"
#include "render/Mask.h"
#include "render/Model.h"
#include "render/RenderContext.h"
#include "tools/Gui.h"
#include "tools/Handeye.h"
#include "tools/PoseLog.h"
#include "tools/ProgressBar.hpp"

#define FPS 29.97

class RenderingModule
{
    public:
        RenderingModule(int argc, char* argv[]);

        virtual ~RenderingModule(void);

        void launch(void);

    private:
        void loadParams(std::string filepath);
        
        unsigned int getFrameCount(std::string directoryPath);

        void writeOBJ(const std::string &filename,const Model *model,const uint8_t *coverageTex, glm::mat4 modelTransform);

        Mask            *mask;
        Handeye         *handeye;
        Model           *model;
        PoseLog         *poseLog;
        progressbar     *bar;
        RenderContext   *context;

        glm::mat4 T_final;

        unsigned int numFrames;
        
        /* Omnidirectional camera intrinsics. */
        unsigned int width, height;
        float cx, cy;
        float a0, a1, a2, a3, a4;
        float c, d, e;

        /* Handeye camera transforms. */
        glm::mat4 A_cal;
        glm::mat4 B_cal;
        glm::mat4 X;

        /* File paths. */
        std::string modelFilePath;
        std::string poseFilePath;
        std::string rgbFolderPath;
        std::string maskFilePath;
        std::string renderFolderPath;

        /* Config values. */
        std::vector<float> modelTransformR6;
        float poseStartTime;

        /* Image w/ no alpha device memory. */
        uint16_t *normalsNoAlpha_dev;
        uint16_t *flowNoAlpha_dev;

        /* Image host memory. */
        uint16_t *depth_host;
        uint16_t *normals_host;
        uint16_t *flow_host;
        uint8_t  *occlusion_host;
        uint8_t  *coverage_host;
};
#endif /* RENDERINGMODULE_H_ */
