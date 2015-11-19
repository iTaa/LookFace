#
#
#

OPENCV = `pkg-config --cflags --libs opencv`

#
#
#

output:
	$(CXX) sample.cpp ../picornt.c ../picornt.h -I../ -lrt $(OPENCV) -o pico -O3 -std=c++11
