/***********************************************************************************/
/*
 *	File name:	Intrinsics.h
 *
 *	Author:     Taylor Bobrow, Johns Hopkins University (2023)
 * 
 */

#ifndef INTRINSICS_H_
#define INTRINSICS_H_

#include <glm/glm.hpp>

/*  Omnidirectional camera intrinsics from
    Scaramuzza, D., A. Martinelli, and R. Siegwart.
    "A Toolbox for Easy Calibrating Omnidirectional Cameras."
    Proceedings to IEEE International Conference on
    Intelligent Robots and Systems, (IROS). Beijing, China, 
    October 7–15, 2006.*/
struct Intrinsics
{
    glm::ivec2  size;
    glm::vec2   center;
    glm::vec4   polyCoeff;
    glm::mat2   stretchMat;

    Intrinsics( unsigned int w, unsigned int h,
                float cx, float cy,
                float a0, float a1, float a2, float a3, float a4,
                float c, float d, float e)
    {
        size        = glm::ivec2(w,h);
        center      = glm::vec2(cx,cy);
        polyCoeff   = glm::vec4(a0,a2,a3,a4);
        stretchMat  = glm::mat2(c,d,e,1.0);
    }
};

#endif /* INTRINSICS_H_*/