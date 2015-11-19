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

include_directories(${BOOST_INCLUDE_DIR})

SET(SOURCE
    src/utils.cpp
    src/regressor.cpp
    src/randomforest.cpp
    src/rlbf.cpp
    includes/liblinear/blas/daxpy.c
    includes/liblinear/blas/ddot.c
    includes/liblinear/blas/dnrm2.c
    includes/liblinear/blas/dscal.c
    includes/liblinear/tron.cpp
    includes/liblinear/linear.cpp
)

SET(HEADERS
    includes/headers.h
    includes/utils.h
    includes/regressor.h
    includes/randomforest.h
    includes/rlbf.h
    includes/liblinear/blas/blas.h
    includes/liblinear/blas/blasp.h
    includes/liblinear/tron.h
    includes/liblinear/linear.h
)

# link
link_directories(/usr/local/lib)
link_directories(lib/)

include_directories(./includes)
include_directories(../pico/includes)
include_directories(/usr/include/libxml2)

add_library( FaceRlbf SHARED ${SOURCE} ${HEADERS})

install (TARGETS FaceRlbf DESTINATION bin)
install (FILES HEADERS DESTINATION include)
