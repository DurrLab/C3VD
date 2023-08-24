/***********************************************************************************/
/*
 *	File name:	Gui.h
 *
 *	Author:     Taylor Bobrow, Johns Hopkins University (2023)
 * 
 */

#ifndef GUI_H_
#define GUI_H_

#include <pangolin/pangolin.h>
#include <pangolin/display/display.h>
#include <pangolin/display/view.h>
#include <pangolin/handler/handler.h>
#include <pangolin/gl/gldraw.h>
#include <pangolin/gl/glcuda.h>

#include "../render/RenderFlags.h"

#define MAX_ROTATION_STEP       0.5
#define MAX_TRANSLATION_STEP    25
#define PREVIEW_HEIGHT          250
#define UI_WIDTH                250

class Gui
{
    public:
        Gui(unsigned int width, unsigned int height);

        ~Gui(void);

        void render(void);

        void copyToDisplayBuffer(const void* fbptr, uint8_t flag);

        void loadTargetImg(std::string imgpath);

        void printKeyInputs(void);

        pangolin::Var<float>    *modelT_roll;
        pangolin::Var<float>    *modelT_pitch;
        pangolin::Var<float>    *modelT_yaw;
        pangolin::Var<float>    *modelT_x;
        pangolin::Var<float>    *modelT_y;
        pangolin::Var<float>    *modelT_z;
        pangolin::Var<float>    *stepVelocity;
        pangolin::Var<float>    *poseStartTime;
        pangolin::Var<int>      *currentFrame;
        pangolin::Var<bool>     *overlayTarget;
        pangolin::Var<bool>     *previewDepth;
        pangolin::Var<bool>     *previewNormals;
        pangolin::Var<bool>     *previewFlow;
        pangolin::Var<bool>     *previewOcclusion;
        pangolin::Var<float>    *overlayAlpha;
    
    private:

        float checkRotationBounds(float angle);

        float clamp(float value, float lbound, float ubound);

        const unsigned int width, height;

        pangolin::Params inputs;

        pangolin::GlTextureCudaArray *diffuseTex;
        pangolin::GlTextureCudaArray *depthTex;
        pangolin::GlTextureCudaArray *normalsTex;
        pangolin::GlTextureCudaArray *flowTex;
        pangolin::GlTextureCudaArray *occlusionTex;
        pangolin::GlTexture *targetTex;

        pangolin::GlSlProgram       blendShaderProg;
        pangolin::GlFramebuffer     *fboBlend;
        pangolin::GlTexture         *texBlend;
        pangolin::GlRenderBuffer    *renderBlend;
};
#endif /* GUI_H_ */
