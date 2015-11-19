# License Agreement (3-clause BSD License)
# Copyright (c) 2015, Klaus Haag, all rights reserved.
# Third party copyrights and patents are property of their respective owners.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the names of the copyright holders nor the names of the contributors
#   may be used to endorse or promote products derived from this software
#   without specific prior written permission.
#
# This software is provided by the copyright holders and contributors "as is" and
# any express or implied warranties, including, but not limited to, the implied
# warranties of merchantability and fitness for a particular purpose are disclaimed.
# In no event shall copyright holders or contributors be liable for any direct,
# indirect, incidental, special, exemplary, or consequential damages
# (including, but not limited to, procurement of substitute goods or services;
# loss of use, data, or profits; or business interruption) however caused
# and on any theory of liability, whether in contract, strict liability,
# or tort (including negligence or otherwise) arising in any way out of
# the use of this software, even if advised of the possibility of such damage.

project(CfTracking)

cmake_minimum_required(VERSION 2.8.7)

# option to enable OpenMP; only relevant for the KCF version with the
# VOT scale estimation
option(WITH_OPENMP "Enable OpenMP" OFF)

if(WITH_OPENMP)
    find_package(OpenMP REQUIRED)

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif(WITH_OPENMP)

# add c++11 support
if(CMAKE_COMPILER_IS_GNUCC)
    ADD_DEFINITIONS ( -std=c++11 )
endif(CMAKE_COMPILER_IS_GNUCC)

# add OpenCV
set(OPENCV_DIR_HINT "")

if(WIN32)
    get_filename_component(OPENCV_DIR_PLATFORM $ENV{OPENCV_DIR} DIRECTORY)
    get_filename_component(OPENCV_DIR_HINT ${OPENCV_DIR_PLATFORM} DIRECTORY)
endif(WIN32)

set(OpenCV_STATIC OFF)
find_package(OpenCV REQUIRED HINTS ${OPENCV_DIR_HINT})

# add 3rdparty source
set(CF_TCLAP_DIR "../includes/3rdparty/tclap")
set(CF_CV_EXT_DIR "../includes/3rdparty/cv_ext")
set(CF_PIOTR_DIR "../includes/3rdparty/piotr")

include_directories(${OpenCV_DIR}/include)
include_directories(${CF_TCLAP_DIR}
                    ${CF_CV_EXT_DIR}
                    ${CF_PIOTR_DIR}
                    ${CF_PIOTR_DIR}/src
                    ../includes
                    ../includes/cf_libs)

include_directories(../includes/cf_libs/kcf
                    ../includes/cf_libs/dsst
                    ../includes/cf_libs/common)

link_directories(/usr/local/lib
                ./lib)                    

add_executable(DSSTcpp
    main_dsst.cpp
    ../includes/cf_libs/dsst/dsst_debug.hpp
    ../includes/cf_libs/dsst/dsst_tracker.hpp)
target_link_libraries(DSSTcpp DsstTrack ${OpenCV_LIBS})

add_executable(KCFcpp
    main_kcf.cpp
    ../includes/cf_libs/kcf/kcf_debug.hpp
    ../includes/cf_libs/kcf/kcf_tracker.hpp)
target_link_libraries(KCFcpp KcfTrack ${OpenCV_LIBS})

