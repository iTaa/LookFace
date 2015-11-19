# RUN for C++ 11
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
        message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

set(CF_TCLAP_DIR "../../3rdparty/tclap")
set(CF_CV_EXT_DIR "../../3rdparty/cv_ext")
set(CF_PIOTR_DIR "../../3rdparty/piotr")

include_directories(${BOOST_INCLUDE_DIR})
include_directories(/usr/include/libxml2)
include_directories(${CF_TCLAP_DIR}
                    ${CF_CV_EXT_DIR}
                    ${CF_PIOTR_DIR}
                    ${CF_PIOTR_DIR}/src
                    ../common
                    ../kcf
                    ../dsst
                    ../)

set(CF_PIOTR_SOURCES
    ${CF_PIOTR_DIR}/gradientMex.hpp
    ${CF_PIOTR_DIR}/src/gradientMex.cpp
    ${CF_PIOTR_DIR}/src/sse.hpp
    ${CF_PIOTR_DIR}/src/wrappers.hpp)

set(CF_LIB_COMMON_SOURCES
    ../common/feature_channels.hpp
    ../common/mat_consts.hpp
    ../common/math_helper.hpp
    ../common/math_helper.cpp
    ../common/cf_tracker.hpp
    ../common/tracker_debug.hpp
    ../common/scale_estimator.hpp
    ../common/cv_ext.hpp
    ${CF_CV_EXT_DIR}/shift.cpp
    ${CF_CV_EXT_DIR}/shift.hpp
    ${CF_CV_EXT_DIR}/math_spectrums.cpp
    ${CF_CV_EXT_DIR}/math_spectrums.hpp
    ${CF_CV_EXT_DIR}/psr.hpp
    ${CF_PIOTR_SOURCES})

set(CF_MAIN_SOURCES
    ${CF_CV_EXT_DIR}/init_box_selector.hpp
    ${CF_CV_EXT_DIR}/init_box_selector.cpp
    ${CF_CV_EXT_DIR}/tracker_run.hpp
    ${CF_CV_EXT_DIR}/tracker_run.cpp
    ${CF_TCLAP_DIR}/tclap/CmdLine.h
    ../image_acquisition.hpp
    ../image_acquisition.cpp
    ../dsst/dsst_debug.hpp
    ../dsst/dsst_tracker.hpp
    kcf_debug.hpp
    kcf_tracker.hpp)

# link
link_directories(/usr/local/lib)

add_library( KcfTrack SHARED ${CF_PIOTR_SOURCES} ${CF_LIB_COMMON_SOURCES} ${CF_MAIN_SOURCES})

install (TARGETS KcfTrack DESTINATION bin)
install (FILES HEADERS DESTINATION include)
