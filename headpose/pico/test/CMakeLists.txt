cmake_minimum_required (VERSION 2.6)

project (test)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/modules/")

# OpenCV
if(UNIX)
    find_package( OpenCV 3.0 REQUIRED )
    find_package( Boost REQUIRED COMPONENTS filesystem system)
    MESSAGE("Boost information:")
    MESSAGE("  Boost_INCLUDE_DIRS: ${Boost_INCLUDE_DIRS}")
    MESSAGE("  Boost_LIBRARIES: ${Boost_LIBRARIES}")
    MESSAGE("  Boost_LIBRARY_DIRS: ${Boost_LIBRARY_DIRS}")

    INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})
    INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS}/boost)
    LINK_DIRECTORIES(${Boost_LIBRARY_DIRS})
endif(UNIX)

# Crosss platform
option(Platform "Platform" X64)
# Find out what system we use to include the needed libs
if(Platform MATCHES X64) # x64-bit
    MESSAGE ("Configure Platform To X64")
    if (${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
        execute_process(COMMAND ${CMAKE_CXX_COMPILER} -dumpversion OUTPUT_VARIABLE GCC_VERSION)
        if (GCC_VERSION VERSION_LESS 4.7)
            set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x -msse -msse2 -msse3")
        else ()
            set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -msse -msse2 -msse3")
        endif ()
    else ()
        set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -msse -msse2 -msse3")
    endif ()
else()
    MESSAGE ("Configure Platform To ARM")
    if (${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
        execute_process(COMMAND ${CMAKE_CXX_COMPILER} -dumpversion OUTPUT_VARIABLE GCC_VERSION)
        if (GCC_VERSION VERSION_LESS 4.7)
            set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x -march=native -mtune=cortex-a15 -mfpu=neon")
        else ()
            set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -march=native -mtune=cortex-a15 -mfpu=neon")
        endif ()
    else ()
        set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -march=native -mtune=cortex-a15 -mfpu=neon")
    endif ()
endif()

# Boost
if(WIN32)
    include_directories(lib/3rdParty/boost)
    include_directories(lib/3rdParty/boost/boost)
    link_directories( ${PROJECT_SOURCE_DIR}/lib/3rdParty/boost/lib )
else()
    INCLUDE_DIRECTORIES(${BOOST_INCLUDE_DIR})
endif()

# Include
include_directories(../)
include_directories(../includes)
include_directories(/usr/include/libxml2)

# Local libraries
include_directories(../inlcudes)

# link
link_directories(/usr/local/lib)
link_directories( ${PROJECT_SOURCE_DIR}/lib/)

add_executable(test sample.cpp)

target_link_libraries(test FacePico ${OpenCV_LIBS} ${Boost_LIBRARIES})
