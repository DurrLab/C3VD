/***********************************************************************************/
/*
 *	File name:	Rgba2r.cuh
 *
 *	Author:     Taylor Bobrow, Johns Hopkins University (2023)
 * 
 */

#ifndef RGBA2R_H_
#define RGBA2R_H_
      
template <typename T> void rgba2R(T* rgba_dev, T* r_dev, const unsigned int width, const unsigned int height);

#endif /* RGBA2R_H_ */