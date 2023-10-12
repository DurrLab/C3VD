cmake_minimum_required(VERSION 3.22)

project(c3vd LANGUAGES C CXX CUDA)

# set binary output dir
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin/)

# external paths
set(GLM_ROOT_DIR        ${PROJECT_SOURCE_DIR}/external/glm)
set(OptiX_ROOT_DIR      ${PROJECT_SOURCE_DIR}/external/optix)
set(OWL_SOURCE_DIR      ${PROJECT_SOURCE_DIR}/external/owl)
set(PANGOLIN_SOURCE_DIR ${PROJECT_SOURCE_DIR}/external/pangolin)
set(TINYTIFF_SOURCE_DIR ${PROJECT_SOURCE_DIR}/external/tinytiff)

# OpenGL
find_package(OpenGL)

# Owl/Optix
add_subdirectory(${OWL_SOURCE_DIR} EXCLUDE_FROM_ALL)

# Pangolin
set(BUILD_TOOLS OFF CACHE INTERNAL "" FORCE)
set(BUILD_EXAMPLES OFF CACHE INTERNAL "" FORCE)
set(BUILD_PANGOLIN_VIDEO OFF CACHE INTERNAL "" FORCE)
add_subdirectory(${PANGOLIN_SOURCE_DIR})
include_directories(SYSTEM ${PANGOLIN_SOURCE_DIR}/include)

# GLM
include_directories(SYSTEM ${GLM_ROOT_DIR})

# libtiff
set(TinyTIFF_BUILD_TESTS OFF CACHE INTERNAL "" FORCE)
add_subdirectory(${TINYTIFF_SOURCE_DIR})

# Create ptx code
embed_ptx(
    OUTPUT_TARGET
        render-ptx
    EMBEDDED_SYMBOL_NAMES
        ptxCode
    PTX_LINK_LIBRARIES
        owl::owl
    SOURCES
        render/Render.cu
)

# Render
set(RENDER_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/render/Float3.h
    ${CMAKE_CURRENT_SOURCE_DIR}/render/Intrinsics.h
    ${CMAKE_CURRENT_SOURCE_DIR}/render/LaunchParams.h
    ${CMAKE_CURRENT_SOURCE_DIR}/render/Mask.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/render/Mask.h
    ${CMAKE_CURRENT_SOURCE_DIR}/render/Model.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/render/Model.h
    ${CMAKE_CURRENT_SOURCE_DIR}/render/Quartic.cuh
    ${CMAKE_CURRENT_SOURCE_DIR}/render/RenderContext.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/render/RenderContext.h
    ${CMAKE_CURRENT_SOURCE_DIR}/render/RenderFlags.h
    ${CMAKE_CURRENT_SOURCE_DIR}/render/Rgba2rgb.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/render/Rgba2rgb.cuh    
    ${CMAKE_CURRENT_SOURCE_DIR}/render/tiny_obj_loader.h
    ${CMAKE_CURRENT_SOURCE_DIR}/render/TransformFlags.h
)

# Tools
set(TOOLS_SRC 
    ${CMAKE_CURRENT_SOURCE_DIR}/tools/ConfigParser.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tools/ConfigParser.h
    ${CMAKE_CURRENT_SOURCE_DIR}/tools/Gui.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tools/Gui.h
    ${CMAKE_CURRENT_SOURCE_DIR}/tools/Handeye.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tools/Handeye.h
    ${CMAKE_CURRENT_SOURCE_DIR}/tools/PoseLog.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tools/PoseLog.h
    ${CMAKE_CURRENT_SOURCE_DIR}/tools/ProgressBar.hpp
)

add_executable( c3vd
                Main.cpp
                AlignmentModule.cpp
                AlignmentModule.h
                RegistrationModule.cpp
                RegistrationModule.h
                RenderingModule.cpp
                RenderingModule.h
                ${RENDER_SRC}
                ${TOOLS_SRC}
)

target_link_libraries( c3vd PRIVATE
                       render-ptx
                       owl::owl
                       stb_image
                       pangolin
                       TinyTIFFShared
)
