/***********************************************************************************/
/*
 *	File name:	Gui.cpp
 *
 *	Author:     Taylor Bobrow, Johns Hopkins University (2023)
 * 
 */

#include "Gui.h"

Gui::Gui(const unsigned int width, const unsigned int height)
: width(width),
  height(height)
{   
    pangolin::CreateWindowAndBind("C3VD", UI_WIDTH+width, PREVIEW_HEIGHT+height, inputs);

    /* Setup menu panel for UI input. */
    pangolin::CreatePanel("ui").SetBounds(0.0,1.0,0.0,pangolin::Attach::Pix(UI_WIDTH));
    modelT_roll         = new pangolin::Var<float>("ui.Roll (rad)", M_PI, 0.0, 2*M_PI);
    modelT_pitch        = new pangolin::Var<float>("ui.Pitch (rad)", M_PI, 0.0, 2*M_PI);
    modelT_yaw          = new pangolin::Var<float>("ui.Yaw (rad)", M_PI, 0.0, 2*M_PI);
    modelT_x            = new pangolin::Var<float>("ui.tx (mm)", 0.0, -200.0, 200.0);
    modelT_y            = new pangolin::Var<float>("ui.ty (mm)", 0.0, -200.0, 200.0);
    modelT_z            = new pangolin::Var<float>("ui.tz (mm)", 0.0, -200.0, 200.0);
    stepVelocity        = new pangolin::Var<float>("ui.Step velocity (%)", 20.0, 0.0, 100.0);
    poseStartTime       = new pangolin::Var<float>("ui.Pose start time (sec)", 0.0, 0.0, 30.0);
    currentFrame        = new pangolin::Var<int>("ui.Current frame", 0, 0, 100);
    overlayAlpha        = new pangolin::Var<float>("ui.Alpha overlay", 0.0, 0.0, 100.0);
    overlayTarget       = new pangolin::Var<bool>("ui.Overlay target frames", false, true);
    previewDepth        = new pangolin::Var<bool>("ui.Preview depth", false, true);
    previewNormals      = new pangolin::Var<bool>("ui.Preview surface normals", false, true);
    previewFlow         = new pangolin::Var<bool>("ui.Preview optical flow", false, true);
    previewOcclusion    = new pangolin::Var<bool>("ui.Preview occlusion", false, true);

    /* Keyboard callback functions. */
    pangolin::RegisterKeyPressCallback('q', [&](){
        modelT_roll->Ref().Set(checkRotationBounds(modelT_roll->Get()+(stepVelocity->Get()/100.f*(float)MAX_ROTATION_STEP)));
    });
    pangolin::RegisterKeyPressCallback('a', [&](){
        modelT_roll->Ref().Set(checkRotationBounds(modelT_roll->Get()-(stepVelocity->Get()/100.f*(float)MAX_ROTATION_STEP)));
    });
    pangolin::RegisterKeyPressCallback('w', [&](){
        modelT_pitch->Ref().Set(checkRotationBounds(modelT_pitch->Get()+(stepVelocity->Get()/100.f*(float)MAX_ROTATION_STEP)));
    });
    pangolin::RegisterKeyPressCallback('s', [&](){
        modelT_pitch->Ref().Set(checkRotationBounds(modelT_pitch->Get()-(stepVelocity->Get()/100.f*(float)MAX_ROTATION_STEP)));
    });
    pangolin::RegisterKeyPressCallback('e', [&](){
        modelT_yaw->Ref().Set(checkRotationBounds(modelT_yaw->Get()+(stepVelocity->Get()/100.f*(float)MAX_ROTATION_STEP)));
    });
    pangolin::RegisterKeyPressCallback('d', [&](){
        modelT_yaw->Ref().Set(checkRotationBounds(modelT_yaw->Get()-(stepVelocity->Get()/100.f*(float)MAX_ROTATION_STEP)));
    });
    pangolin::RegisterKeyPressCallback('r', [&](){
        modelT_x->Ref().Set( clamp(modelT_x->Get() + (stepVelocity->Get()/100.f*(float)MAX_TRANSLATION_STEP),
                modelT_x->Ref().Meta().range[0],
                modelT_x->Ref().Meta().range[1]));
    });
    pangolin::RegisterKeyPressCallback('f', [&](){
        modelT_x->Ref().Set(clamp(modelT_x->Get() - (stepVelocity->Get()/100.f*(float)MAX_TRANSLATION_STEP),
                modelT_x->Ref().Meta().range[0],
                modelT_x->Ref().Meta().range[1]));
    });
    pangolin::RegisterKeyPressCallback('t', [&](){
        modelT_y->Ref().Set(clamp(modelT_y->Get() + (stepVelocity->Get()/100.f*(float)MAX_TRANSLATION_STEP),
                modelT_y->Ref().Meta().range[0],
                modelT_y->Ref().Meta().range[1]));
    });
    pangolin::RegisterKeyPressCallback('g', [&](){
        modelT_y->Ref().Set(clamp(modelT_y->Get() - (stepVelocity->Get()/100.f*(float)MAX_TRANSLATION_STEP),
                modelT_y->Ref().Meta().range[0],
                modelT_y->Ref().Meta().range[1]));
    });
    pangolin::RegisterKeyPressCallback('y', [&](){
        modelT_z->Ref().Set(clamp(modelT_z->Get() + (stepVelocity->Get()/100.f*(float)MAX_TRANSLATION_STEP),
                modelT_z->Ref().Meta().range[0],
                modelT_z->Ref().Meta().range[1]));
    });
    pangolin::RegisterKeyPressCallback('h', [&](){
        modelT_z->Ref().Set(clamp(modelT_z->Get() - (stepVelocity->Get()/100.f*(float)MAX_TRANSLATION_STEP),
                modelT_z->Ref().Meta().range[0],
                modelT_z->Ref().Meta().range[1]));
    });
    pangolin::RegisterKeyPressCallback(pangolin::PANGO_SPECIAL + pangolin::PANGO_KEY_UP, [&](){
        poseStartTime->Ref().Set(clamp(poseStartTime->Get() + 0.05, poseStartTime->Ref().Meta().range[0], poseStartTime->Ref().Meta().range[1]));
    });
    pangolin::RegisterKeyPressCallback(pangolin::PANGO_SPECIAL + pangolin::PANGO_KEY_DOWN, [&](){
        poseStartTime->Ref().Set(clamp(poseStartTime->Get() - 0.05, poseStartTime->Ref().Meta().range[0], poseStartTime->Ref().Meta().range[1]));
    });
    pangolin::RegisterKeyPressCallback(pangolin::PANGO_SPECIAL + pangolin::PANGO_KEY_RIGHT, [&](){
        currentFrame->Ref().Set(clamp(currentFrame->Get() + 1, currentFrame->Ref().Meta().range[0], currentFrame->Ref().Meta().range[1]));
    });
    pangolin::RegisterKeyPressCallback(pangolin::PANGO_SPECIAL + pangolin::PANGO_KEY_LEFT, [&](){
        currentFrame->Ref().Set(clamp(currentFrame->Get() - 1, currentFrame->Ref().Meta().range[0], currentFrame->Ref().Meta().range[1]));
    });
    pangolin::RegisterKeyPressCallback('.', [&](){
        overlayAlpha->Ref().Set(clamp(overlayAlpha->Get() + 10, overlayAlpha->Ref().Meta().range[0], overlayAlpha->Ref().Meta().range[1]));
    });
    pangolin::RegisterKeyPressCallback(',', [&](){
        overlayAlpha->Ref().Set(clamp(overlayAlpha->Get() - 10, overlayAlpha->Ref().Meta().range[0], overlayAlpha->Ref().Meta().range[1]));
    });
    pangolin::RegisterKeyPressCallback('i', [&](){
        printKeyInputs();
    });

    /* Initialize gl textures. */
    diffuseTex = new pangolin::GlTextureCudaArray(width, height, GL_RGBA8, false, 0, GL_RGBA, GL_UNSIGNED_BYTE);
    targetTex = new pangolin::GlTextureCudaArray(width, height, GL_RGBA8, false, 0, GL_RGBA, GL_UNSIGNED_BYTE);
    depthTex = new pangolin::GlTextureCudaArray(width, height, GL_LUMINANCE16, false, 0, GL_LUMINANCE, GL_UNSIGNED_SHORT);
    normalsTex = new pangolin::GlTextureCudaArray(width, height, GL_RGBA16, false, 0, GL_RGBA, GL_UNSIGNED_SHORT);
    flowTex = new pangolin::GlTextureCudaArray(width, height, GL_RGBA16, false, 0, GL_RGBA, GL_UNSIGNED_SHORT);
    occlusionTex = new pangolin::GlTextureCudaArray(width, height, GL_LUMINANCE8, false, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE);

    /* Create primary view window for displaying diffuse and target images. */
    pangolin::Display("primary").SetBounds(pangolin::Attach::Pix(PREVIEW_HEIGHT),1.0,pangolin::Attach::Pix(UI_WIDTH),1.0);

    /* Create view windows for previewing rendered ground truth frames. */
    pangolin::Display("preview")
        .SetLayout(pangolin::LayoutEqualHorizontal)
        .SetBounds(pangolin::Attach::Pix(0),pangolin::Attach::Pix(PREVIEW_HEIGHT),pangolin::Attach::Pix(UI_WIDTH),1.0)
        .AddDisplay(pangolin::Display("depth"))
        .AddDisplay(pangolin::Display("normals"))
        .AddDisplay(pangolin::Display("flow"))
        .AddDisplay(pangolin::Display("occlusion"));

    /*  Opengl shaders for blending the real target and rendered
        diffuse frames. */
    const char* gs =
    "#version 330 core\n"
    "layout(points) in;"
    "layout(triangle_strip, max_vertices = 4) out;"
    "out vec2 texcoord;"
    "void main()"
    "{"
    "    gl_Position = vec4( 1.0, 1.0, 0.5, 1.0 );"
    "    texcoord = vec2( 1.0, 1.0 );"
    "    EmitVertex();"
    "    gl_Position = vec4(-1.0, 1.0, 0.5, 1.0 );"
    "    texcoord = vec2( 0.0, 1.0 );"
    "    EmitVertex();"
    "    gl_Position = vec4( 1.0,-1.0, 0.5, 1.0 );"
    "    texcoord = vec2( 1.0, 0.0 );"
    "    EmitVertex();"
    "    gl_Position = vec4(-1.0,-1.0, 0.5, 1.0 );"
    "    texcoord = vec2( 0.0, 0.0 );"
    "    EmitVertex();"
    "    EndPrimitive();"
    "}";
    const char* vs =
    "#version 330 core\n"
    "void main()"
    "{"
    "}";
    const char* fs =
        "#version 330 core\n"
        "in vec2 texcoord;"
        "out vec4 FragColor;"
        "uniform sampler2D texA;"
        "uniform sampler2D texB;"
        "uniform float alpha;"
        "void main() {"
        "   vec2 uv = texcoord;"
        "   if(0.0 <= uv.x && uv.x <= 1.0 && 0.0 <= uv.y && uv.y <= 1.0) {"
        "       vec3 valA = texture2D(texA, uv.xy).xyz;"
        "       vec3 valB = texture2D(texB, uv.xy).xyz;"
        "       FragColor = vec4(mix(valA,valB,alpha),1.0);"
        "   }"
        "   else {"
        "       FragColor = vec4(0.0,0.0,0.0,0.0);"
        "   }"
        "}";
    blendShaderProg.AddShader( pangolin::GlSlGeometryShader, gs );
    blendShaderProg.AddShader( pangolin::GlSlVertexShader,   vs );
    blendShaderProg.AddShader( pangolin::GlSlFragmentShader, fs );
    blendShaderProg.Link();

    /* Frame buffer object for rendering the blended images. */
    renderBlend = new pangolin::GlRenderBuffer(width,height);
    texBlend    = new pangolin::GlTexture(width, height, GL_RGBA8, true, 0, GL_RGBA, GL_UNSIGNED_BYTE);
    fboBlend    = new pangolin::GlFramebuffer;
    fboBlend->AttachColour(*texBlend);
    fboBlend->AttachDepth(*renderBlend);
}

Gui::~Gui(void)
{
    delete modelT_roll;
    delete modelT_pitch;
    delete modelT_yaw;
    delete modelT_x;
    delete modelT_y;
    delete modelT_z;
    delete stepVelocity;
    delete poseStartTime;
    delete currentFrame;
    delete overlayTarget;
    delete overlayAlpha;
    delete previewDepth;
    delete previewNormals;
    delete previewFlow;
    delete previewOcclusion;
    delete diffuseTex;
    delete depthTex;
    delete normalsTex;
    delete flowTex;
    delete occlusionTex;
    delete targetTex;
    delete fboBlend;
    delete texBlend;
    delete renderBlend;
}

/* Called once per render cycle to update the GUI display. */
void Gui::render(void)
{
    /* Blend the target and diffuse images. */
    fboBlend->Bind();
    {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glViewport(0,0,width,height);

        blendShaderProg.Bind();

        blendShaderProg.SetUniform("texA", 0);
        blendShaderProg.SetUniform("texB", 1);
        blendShaderProg.SetUniform("alpha", overlayAlpha->Get()/100.0f);

        glEnable(GL_TEXTURE_2D);

        glColor4f(1,1,1,1);

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, diffuseTex->tid);

        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, targetTex->tid);

        glDrawArrays(GL_POINTS, 0, 1);
        
        glDisable(GL_TEXTURE_2D);

        glBindTexture(GL_TEXTURE_2D, 0);

        glActiveTexture(GL_TEXTURE0);

        blendShaderProg.Unbind();
    }
    fboBlend->Unbind();

    /* Update all of the view windows with the latest texture memory. */
    pangolin::Display("primary").Activate();
    glColor4f(1.0f,1.0f,1.0f,1.0f);
    texBlend->RenderToViewportFlipY();

    pangolin::Display("depth").Activate();
    glColor4f(1.0f,1.0f,1.0f,1.0f);
    depthTex->RenderToViewportFlipY();

    pangolin::Display("normals").Activate();
    glColor4f(1.0f,1.0f,1.0f,1.0f);
    normalsTex->RenderToViewportFlipY();

    pangolin::Display("flow").Activate();
    glColor4f(1.0f,1.0f,1.0f,1.0f);
    flowTex->RenderToViewportFlipY();

    pangolin::Display("occlusion").Activate();
    glColor4f(1.0f,1.0f,1.0f,1.0f);
    occlusionTex->RenderToViewportFlipY();

    pangolin::FinishFrame();
}

/*  Copy the input CUDA memory holding the rendered frame into the OpenGL
    textures for display.
*/
void Gui::copyToDisplayBuffer(const void* fbptr, uint8_t flag)
{
    if(flag & RenderFlags::DIFFUSE)
        pangolin::CopyDevMemtoTex((uint32_t*)fbptr, width * sizeof(uint32_t), *diffuseTex);
    if(flag & RenderFlags::DEPTH)
        pangolin::CopyDevMemtoTex((uint16_t*)fbptr, width * sizeof(uint16_t), *depthTex);
    if(flag & RenderFlags::SCREEN_SPACE_NORMALS)
        pangolin::CopyDevMemtoTex((ushort4*)fbptr, width * sizeof(ushort4), *normalsTex);
    if(flag & RenderFlags::OPTICAL_FLOW)
        pangolin::CopyDevMemtoTex((ushort4*)fbptr, width * sizeof(ushort4), *flowTex);
    if(flag & RenderFlags::OCCLUSION)
        pangolin::CopyDevMemtoTex((uint8_t*)fbptr, width * sizeof(uint8_t), *occlusionTex);
}

/*  Read the target image stored at *imgpath. */
void Gui::loadTargetImg(std::string imgpath)
{
    glPixelStorei(GL_UNPACK_ALIGNMENT,1);

    pangolin::TypedImage img_host = pangolin::LoadImage(imgpath);

    targetTex->Upload(img_host.ptr,GL_RGB,GL_UNSIGNED_BYTE);
}

/* Bounds handling for model rotation inputs. */
float Gui::checkRotationBounds(float angle)
{
    if (angle > 2.0*M_PI)   return angle - 2*M_PI;
    else if (angle < 0.0)   return 2*M_PI + angle;
    else                    return angle;
}

/* Bounds handling for other user inputs. */
float Gui::clamp(float value, float lbound, float ubound)
{
    if (value > ubound)         return ubound;
    else if (value < lbound)    return lbound;
    else                        return value;
}

/* Print a legend of key inputs to the cli. */
void Gui::printKeyInputs(void)
{
      std::cout << "q/a:                increase/decrease the model x-axis rotation angle"  << std::endl;
      std::cout << "w/s:                increase/decrease the model y-axis rotation angle"  << std::endl;
      std::cout << "e/d:                increase/decrease the model z-axis rotation angle"  << std::endl;
      std::cout << "r/f:                increase/decrease the model x-axis translation"     << std::endl;
      std::cout << "t/g:                increase/decrease the model y-axis translation"     << std::endl;
      std::cout << "y/h:                increase/decrease the model z-axis translation"     << std::endl;
      std::cout << "right/left arrow:   increase/decrease the current frame"                << std::endl;
      std::cout << "up/down arrow:      increase/decrease the pose log offset"              << std::endl;
      std::cout << "./,:                increase/decrease the image overlay alpha"          << std::endl;
}