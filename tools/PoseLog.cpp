/***********************************************************************************/
/*
 *	File name:	PoseLog.cpp
 *
 *	Author:     Taylor Bobrow, Johns Hopkins University (2023)
 * 
 */

#include "PoseLog.h"

PoseLog::PoseLog( const std::string &filename)
{
    std::ifstream file;

    file.open(filename.c_str());

    if (!file)
        throw std::runtime_error("Error: could not open pose file " + filename );

    while (!file.eof())
    {
        std::string line;

        std::getline(file, line);

        if(file.eof())
            break;

        float time;

        glm::mat4 T(1.0);

        int n = sscanf(line.c_str(), "%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f",
                        &time, 
                        &T[0][0], &T[1][0], &T[2][0], &T[3][0],
                        &T[0][1], &T[1][1], &T[2][1], &T[3][1],
                        &T[0][2], &T[1][2], &T[2][2], &T[3][2],
                        &T[0][3], &T[1][3], &T[2][3], &T[3][3]);

        if(n != 17)
            throw std::runtime_error( "Error: " + filename + " is incorrectly formatted" );

        trajectory[time] = T;
    }

    file.close();

    /* The file should contain at least one pose. */
    if(trajectory.size() < 1)
        throw std::runtime_error("Error: the file " + filename + " does not contain at least one pose" );

}

/* Returns a pose linearly interpolated pose at t = @timestamp. */
glm::mat4 PoseLog::getTransform(const float &timestamp)
{
    /* Check that the requested time is within bounds. */
    if( !(timestamp >= getBeginTime()  &&  timestamp <= getEndTime()) )
        throw std::runtime_error( "The requsted pose time is not within bounds." );

    /* Lower bound pose. */
    std::map<float, glm::mat4>::const_iterator it0 = trajectory.lower_bound(timestamp);
    float t0 = it0->first;
    glm::mat4 v0 = it0->second;

    /* Upper bound pose. */
    std::map<float, glm::mat4>::const_iterator it1 = trajectory.upper_bound(timestamp);
    float t1 = it1->first;
    glm::mat4 v1 = it1->second;

    /* If entries are the same. */
    if(t0==t1)
        return v0;

    /* Return a weighted linear interpolation of the two closest poses. */
    float w = (timestamp-t0)/(t1-t0);
    glm::quat r0 = glm::quat_cast(v0);
    glm::quat r1 = glm::quat_cast(v1);
    glm::quat rw = glm::slerp(r0,r1,w);
    
    glm::vec4 p0 = glm::column(v0,3);
    glm::vec4 p1 = glm::column(v1,3);
    glm::vec4 pw = glm::mix(p0,p1,w);

    glm::mat4 vw = glm::mat4_cast(rw);
    vw = glm::column(vw,3,pw);

    return vw;
}

float PoseLog::getBeginTime()
{
    std::map<float, glm::mat4>::const_iterator it = trajectory.begin();
    return it->first;
}

float PoseLog::getEndTime()
{
    std::map<float, glm::mat4>::const_iterator it = std::prev(trajectory.end());
    return it->first;
}