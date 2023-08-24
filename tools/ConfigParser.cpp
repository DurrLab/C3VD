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

#include "ConfigParser.h"
#include <fstream>
#include <sstream>
#include <vector>
#include <iostream>
#include <algorithm>
#include <stdexcept>

using namespace std;

ConfigParser::ConfigParser(string path)
: mPathToConfig(path) {

    ifstream configFile;
    configFile.open( mPathToConfig );

    /* Check if file exists*/
    if (!configFile)
    {
        printf("\x1B[31mError opening configuration file \"%s\" \n\x1B[0m", mPathToConfig.c_str());
        exit(EXIT_FAILURE);
    }

    string line;
    string configParamName;
    string rawConfigContent;
    vector<string> configContent;

    size_t lineNbr = 1;

    while( getline(configFile, line) )
    {

        line.erase(std::remove_if( line.begin(), line.end(),
        [l = std::locale {}](auto ch) { return std::isspace(ch, l); }), line.end());

        /* Empty line*/
        if (line.empty()) {
            lineNbr++;
            continue;
        }

        /* Comment line*/
        if (line.at(0) == ';') {
            lineNbr++;
            continue;
        }

        /* Parameter entry*/
        size_t equalSignPos = line.find('=');
        if(equalSignPos != string::npos)
        {   
            /* Parameter name*/
            configParamName = line.substr(0, equalSignPos);
            
            /* Parameter entry*/
            rawConfigContent = line.substr(equalSignPos+1, line.length()-1);

            stringstream ss(rawConfigContent);
            while (ss.good())
            {
                string tmp;
                getline(ss, tmp, ',');
                configContent.push_back(tmp);
            }
        }
        else
        {
            string errorMessage = path + ":" + to_string(lineNbr) + ": parsing error \n" + line;
            throw std::runtime_error(errorMessage);
        }

        mConfigurations.insert(std::make_pair(configParamName, configContent));
        configContent.clear();
        lineNbr++;
    }
}

template <typename T>
T ConfigParser::aConfig(std::string configParamName, size_t pos)
{
    
    if(!doesParamExist(configParamName))
    {
        printf("\x1B[31mParameter \"%s\" is not included in the configuration file\n\x1B[0m", configParamName.c_str());
        exit(EXIT_FAILURE);
    }

    T tmp;
    std::string* config = &mConfigurations[configParamName][pos];
    std::istringstream iss(*config);

    if (config->find( "0x" ) != std::string::npos)
        iss >> std::hex >> tmp;
    else
        iss >> std::dec >> tmp;

    return tmp;
}

template int ConfigParser::aConfig(std::string, size_t);
template float ConfigParser::aConfig(std::string, size_t);
template double ConfigParser::aConfig(std::string, size_t);
template std::string ConfigParser::aConfig(std::string, size_t);
template <>
bool ConfigParser::aConfig<bool>(std::string configParamName, size_t pos)
{   
    if(!doesParamExist(configParamName))
    {
        printf("\x1B[31mParameter \"%s\" is not included in the configuration file\n\x1B[0m", configParamName.c_str());
        exit(EXIT_FAILURE);
    }

    bool tmp;
    std::string config = mConfigurations[configParamName][pos];
    std::istringstream iss(config);

    if (config == "true" ||
        config == "TRUE" ||
        config == "1") {
        return true;
    }

    else if (config == "false" ||
        config == "FALSE" ||
        config == "0") {
        return false;
    }
    else
        return false;
}

template <typename T>
std::vector<T> ConfigParser::aConfigVec(std::string configParamName)
{
    if(!doesParamExist(configParamName))
    {
        printf("\x1B[31mParameter \"%s\" is not included in the configuration file\n\x1B[0m", configParamName.c_str());
        exit(EXIT_FAILURE);
    }

    std::vector<std::string> config;

    config = mConfigurations[configParamName];

    std::vector<T>  tmp(config.size());
    for (unsigned i = 0; i < config.size(); i++)
    {
        std::istringstream iss(config[i]);

        if (config[i].find( "0x" ) != std::string::npos)
            iss >> std::hex >> tmp[i];
        else
            iss >> std::dec >> tmp[i];
    }
    return tmp;
}

template std::vector<int> ConfigParser::aConfigVec(std::string);
template std::vector<float> ConfigParser::aConfigVec(std::string);
template std::vector<double> ConfigParser::aConfigVec(std::string);
template std::vector<std::string> ConfigParser::aConfigVec(std::string);
template <>
std::vector<bool> ConfigParser::aConfigVec<bool>(std::string configParamName)
{
    if(!doesParamExist(configParamName))
    {
        printf("\x1B[31mParameter \"%s\" is not included in the configuration file\n\x1B[0m", configParamName.c_str());
        exit(EXIT_FAILURE);
    }

    std::vector<std::string> config;

    config = mConfigurations[configParamName];

    std::vector<bool>  tmp(config.size());
    for (unsigned i = 0; i < config.size(); i++)
    {

        if (config[i] == "true" ||
            config[i] == "TRUE" ||
            config[i] == "1") {
            tmp[i] = true;
        }

        else if (config[i] == "false" ||
                config[i] == "FALSE" ||
                config[i] == "0") {
                tmp[i] = false;
        }
        else
            tmp[i] = false;
    }
    return tmp;
}

bool ConfigParser::doesParamExist(std::string configParamName)
{
    auto iter = mConfigurations.find(configParamName);

    /* Parameter is not in config file*/
    if ( iter == mConfigurations.end() )
    {  
        return 0;
    }
    return 1;
}