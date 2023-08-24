/***********************************************************************************/
/*
 *	File name:	Mask.h
 *
 *	Author:     Taylor Bobrow, Johns Hopkins University (2023)
 * 
 */

#ifndef MASK_H_
#define MASK_H_

#include <pangolin/image/image_io.h>

class Mask
{
    public:
        Mask(pangolin::TypedImage mask);

        virtual ~Mask(void);
        
        template <typename T> void apply(T* img_dev);

    private:
        bool* mask_dev;

        const unsigned int width, height;
};

#endif /* MASK_H_ */