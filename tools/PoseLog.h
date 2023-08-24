/***********************************************************************************/
/*
 *	File name:	PoseLog.h
 *
 *	Author:     Taylor Bobrow, Johns Hopkins University (2023)
 * 
 */

#ifndef POSELOG_H_
#define POSELOG_H_

#include <fstream>
#include <iostream>
#include <map>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_access.hpp>
#include <glm/gtc/quaternion.hpp>

class PoseLog
{
    public:
        PoseLog( const std::string &filename );

        glm::mat4 getTransform(const float &timestamp);

        float getBeginTime(void);

        float getEndTime(void);
        
    private:
        std::map< float, /* key: time stamp*/
                  glm::mat4, /* value: pose entry*/
                  std::less<float>,/* comparator*/
                  std::allocator<std::pair<const float, glm::mat4>> /* allocator*/
                > trajectory;

};

#endif /* POSELOG_H_ */