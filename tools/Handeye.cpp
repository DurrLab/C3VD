/***********************************************************************************/
/*
 *	File name:	Handeye.cpp
 *
 *	Author:     Taylor Bobrow, Johns Hopkins University (2023)
 * 
 */

#include "Handeye.h"

Handeye::Handeye( glm::mat4 A_cal, /* initial robot pose*/
                  glm::mat4 B_cal, /* initial camera pose*/
                  glm::mat4 X      /* handeye transform*/ )
: A_cal(A_cal),
  B_cal(B_cal),
  X(X)
{}

/* Returns a camera pose for a provided robot 
   pose using the handeye transform*/
glm::mat4 Handeye::A2B(glm::mat4 A)
{ return B_cal * (glm::inverse(X)*((glm::inverse(A_cal)*A)*X)); }