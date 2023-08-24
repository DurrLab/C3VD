// **Adapted from ConfigParser utility by Daniel Zilles Copyright (c) 2018**
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifndef CONFIGPARSER_H_
#define CONFIGPARSER_H_

#include <string>
#include <vector>
#include <map>
#include <iostream>
#include <sstream>

typedef std::map<std::string, std::vector<std::string>> configList; 

class ConfigParser {

  public:
    ConfigParser(std::string path);

    template<typename T>
    T aConfig(std::string configParamName, size_t pos = 0);
    
    template<typename T>
    std::vector<T> aConfigVec(std::string configParamName);

    bool doesParamExist(std::string configParamName);

  private:
    const std::string mPathToConfig;
    configList mConfigurations;

};
#endif /* CONFIGPARSER_H_ */