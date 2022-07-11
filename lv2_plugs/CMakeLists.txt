###############################################
# Standard gumpf
###############################################
cmake_minimum_required (VERSION 3.5)
project (lv2_plugs VERSION 1.0.0)
include (GNUInstallDirs)

add_compile_options (-Wall -pedantic -O3 -ffast-math)

###############################################
# dsp library
###############################################
add_subdirectory (dsp)

###############################################
# build test program
###############################################
add_executable (test_stuff test.c)
target_link_libraries (test_stuff SomaDSP m sndfile)

###############################################
# Add plugin subdirs
###############################################
add_subdirectory (SexyZebra)
