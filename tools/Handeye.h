/***********************************************************************************/
/*
 *	File name:	Handeye.h
 *
 *	Author:     Taylor Bobrow, Johns Hopkins University (2023)
 * 
 */

#ifndef HANDEYE_H_
#define HANDEYE_H_

#include <glm/glm.hpp>

class Handeye
{
    public:
        Handeye(glm::mat4 A_cal, glm::mat4 B_cal, glm::mat4 X);

        glm::mat4 A2B(glm::mat4 A);

    private:
        const glm::mat4 A_cal, 
                        B_cal,
                        X;

};

#endif /* HANDEYE_H_ */