/***********************************************************************************/
/*
 *	File name:	RenderFlags.h
 *
 *	Author:     Taylor Bobrow, Johns Hopkins University (2023)
 *
 */

#ifndef RENDERFLAGS_H_
#define RENDERFLAGS_H_

#include <owl/owl.h>

enum RenderFlags : uint8_t
{ 
    NONE                    = 0,
    DIFFUSE                 = 1 << 1,
    DEPTH                   = 1 << 2,
    SCREEN_SPACE_NORMALS    = 1 << 3,
    OPTICAL_FLOW            = 1 << 4,
    OCCLUSION               = 1 << 5,
    COVERAGE                = 1 << 6,
};

#endif /* RENDERFLAGS_H_ */