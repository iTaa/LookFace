# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.0

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/jxgu/github/NTU-project-lamp/system/lib/tk1/headpose/pico/test

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/jxgu/github/NTU-project-lamp/system/lib/tk1/headpose/pico/test/build

# Include any dependencies generated for this target.
include CMakeFiles/test.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/test.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/test.dir/flags.make

CMakeFiles/test.dir/sample.cpp.o: CMakeFiles/test.dir/flags.make
CMakeFiles/test.dir/sample.cpp.o: ../sample.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/jxgu/github/NTU-project-lamp/system/lib/tk1/headpose/pico/test/build/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/test.dir/sample.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/test.dir/sample.cpp.o -c /home/jxgu/github/NTU-project-lamp/system/lib/tk1/headpose/pico/test/sample.cpp

CMakeFiles/test.dir/sample.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test.dir/sample.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/jxgu/github/NTU-project-lamp/system/lib/tk1/headpose/pico/test/sample.cpp > CMakeFiles/test.dir/sample.cpp.i

CMakeFiles/test.dir/sample.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test.dir/sample.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/jxgu/github/NTU-project-lamp/system/lib/tk1/headpose/pico/test/sample.cpp -o CMakeFiles/test.dir/sample.cpp.s

CMakeFiles/test.dir/sample.cpp.o.requires:
.PHONY : CMakeFiles/test.dir/sample.cpp.o.requires

CMakeFiles/test.dir/sample.cpp.o.provides: CMakeFiles/test.dir/sample.cpp.o.requires
	$(MAKE) -f CMakeFiles/test.dir/build.make CMakeFiles/test.dir/sample.cpp.o.provides.build
.PHONY : CMakeFiles/test.dir/sample.cpp.o.provides

CMakeFiles/test.dir/sample.cpp.o.provides.build: CMakeFiles/test.dir/sample.cpp.o

# Object files for target test
test_OBJECTS = \
"CMakeFiles/test.dir/sample.cpp.o"

# External object files for target test
test_EXTERNAL_OBJECTS =

test: CMakeFiles/test.dir/sample.cpp.o
test: CMakeFiles/test.dir/build.make
test: /usr/local/lib/libopencv_videostab.so.3.0.0
test: /usr/local/lib/libopencv_videoio.so.3.0.0
test: /usr/local/lib/libopencv_video.so.3.0.0
test: /usr/local/lib/libopencv_superres.so.3.0.0
test: /usr/local/lib/libopencv_stitching.so.3.0.0
test: /usr/local/lib/libopencv_shape.so.3.0.0
test: /usr/local/lib/libopencv_photo.so.3.0.0
test: /usr/local/lib/libopencv_objdetect.so.3.0.0
test: /usr/local/lib/libopencv_ml.so.3.0.0
test: /usr/local/lib/libopencv_imgproc.so.3.0.0
test: /usr/local/lib/libopencv_imgcodecs.so.3.0.0
test: /usr/local/lib/libopencv_highgui.so.3.0.0
test: /usr/local/lib/libopencv_hal.a
test: /usr/local/lib/libopencv_flann.so.3.0.0
test: /usr/local/lib/libopencv_features2d.so.3.0.0
test: /usr/local/lib/libopencv_cudev.so.3.0.0
test: /usr/local/lib/libopencv_cudawarping.so.3.0.0
test: /usr/local/lib/libopencv_cudastereo.so.3.0.0
test: /usr/local/lib/libopencv_cudaoptflow.so.3.0.0
test: /usr/local/lib/libopencv_cudaobjdetect.so.3.0.0
test: /usr/local/lib/libopencv_cudalegacy.so.3.0.0
test: /usr/local/lib/libopencv_cudaimgproc.so.3.0.0
test: /usr/local/lib/libopencv_cudafilters.so.3.0.0
test: /usr/local/lib/libopencv_cudafeatures2d.so.3.0.0
test: /usr/local/lib/libopencv_cudacodec.so.3.0.0
test: /usr/local/lib/libopencv_cudabgsegm.so.3.0.0
test: /usr/local/lib/libopencv_cudaarithm.so.3.0.0
test: /usr/local/lib/libopencv_core.so.3.0.0
test: /usr/local/lib/libopencv_calib3d.so.3.0.0
test: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
test: /usr/lib/x86_64-linux-gnu/libboost_system.so
test: /usr/local/lib/libopencv_cudawarping.so.3.0.0
test: /usr/local/lib/libopencv_objdetect.so.3.0.0
test: /usr/local/lib/libopencv_cudafilters.so.3.0.0
test: /usr/local/lib/libopencv_cudaarithm.so.3.0.0
test: /usr/local/lib/libopencv_features2d.so.3.0.0
test: /usr/local/lib/libopencv_ml.so.3.0.0
test: /usr/local/lib/libopencv_highgui.so.3.0.0
test: /usr/local/lib/libopencv_videoio.so.3.0.0
test: /usr/local/lib/libopencv_imgcodecs.so.3.0.0
test: /usr/local/lib/libopencv_flann.so.3.0.0
test: /usr/local/lib/libopencv_video.so.3.0.0
test: /usr/local/lib/libopencv_imgproc.so.3.0.0
test: /usr/local/lib/libopencv_core.so.3.0.0
test: /usr/local/lib/libopencv_cudev.so.3.0.0
test: /usr/local/lib/libopencv_hal.a
test: /usr/local/share/OpenCV/3rdparty/lib/libippicv.a
test: CMakeFiles/test.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable test"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/test.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/test.dir/build: test
.PHONY : CMakeFiles/test.dir/build

CMakeFiles/test.dir/requires: CMakeFiles/test.dir/sample.cpp.o.requires
.PHONY : CMakeFiles/test.dir/requires

CMakeFiles/test.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/test.dir/cmake_clean.cmake
.PHONY : CMakeFiles/test.dir/clean

CMakeFiles/test.dir/depend:
	cd /home/jxgu/github/NTU-project-lamp/system/lib/tk1/headpose/pico/test/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/jxgu/github/NTU-project-lamp/system/lib/tk1/headpose/pico/test /home/jxgu/github/NTU-project-lamp/system/lib/tk1/headpose/pico/test /home/jxgu/github/NTU-project-lamp/system/lib/tk1/headpose/pico/test/build /home/jxgu/github/NTU-project-lamp/system/lib/tk1/headpose/pico/test/build /home/jxgu/github/NTU-project-lamp/system/lib/tk1/headpose/pico/test/build/CMakeFiles/test.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/test.dir/depend

