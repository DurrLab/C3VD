/***********************************************************************************/
/*
 *	File name:	Main.cpp
 *
 *	Author:     Taylor Bobrow, Johns Hopkins University (2023)
 * 
 */

#include <stdexcept>
#include <stdio.h>
#include <string.h>

#include "AlignmentModule.h"
#include "RegistrationModule.h"
#include "RenderingModule.h"

int main(int argc, char* argv[])
{
    if(!strcmp(argv[1],"align"))
    {
        AlignmentModule align(argc, argv);
        align.launch();
    }
    else if(!strcmp(argv[1],"register"))
    {
        RegistrationModule registration(argc, argv);
        registration.launch();
    }  
    else if(!strcmp(argv[1],"rendergt"))
    {
        RenderingModule render(argc, argv);
        render.launch();
    }
    else
    {
        printf("\x1B[31m%s is not a valid program. Valid options are \"align\", \"register\", and \"rendergt\"\n\x1B[0m", argv[1]);
        exit(EXIT_FAILURE);
    }

    return 0;
}