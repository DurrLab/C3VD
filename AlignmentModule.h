/***********************************************************************************/
/*
 *	File name:	AlignmentModule.h
 *
 *	Author:     Taylor Bobrow, Johns Hopkins University (2023)
 * 
 */

#ifndef ALIGNMENTMODULE_H_
#define ALIGNMENTMODULE_H_

#include <dirent.h>
#include <stdio.h>
#include <string>

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <pangolin/image/image_io.h>
#include <pangolin/pangolin.h>

#include "render/LaunchParams.h"
#include "render/Mask.h"
#include "render/Model.h"
#include "render/RenderContext.h"
#include "tools/Gui.h"
#include "tools/Handeye.h"
#include "tools/PoseLog.h"

#define FPS 29.97

class AlignmentModule
{
    public:
        AlignmentModule(int argc, char* argv[]);

        virtual ~AlignmentModule(void);

        void launch(void);

    private:
        void loadParams(std::string filepath);
        
        unsigned int getFrameCount(std::string directoryPath);

    private:
        Gui             *gui;
        Handeye         *handeye;
        Mask            *mask;
        Model           *model;
        PoseLog         *poseLog;
        RenderContext   *context;

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

        /* Config values. */
        std::vector<float> modelTransformR6;
        float poseStartTime;
};
#endif /* ALIGNMENTMODULE_H_ */
