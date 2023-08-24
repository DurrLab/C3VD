/***********************************************************************************/
/*
 *	File name:	RegistrationModule.h
 *
 *	Author:     Taylor Bobrow, Johns Hopkins University (2023)
 * 
 */

#ifndef REGISTRATIONMODULE_H_
#define REGISTRATIONMODULE_H_

class RegistrationModule
{
    public:
        RegistrationModule(int argc, char* argv[]);

        virtual ~RegistrationModule(void);

        void launch(void);

    private:
};
#endif /* REGISTRATIONMODULE_H_ */
