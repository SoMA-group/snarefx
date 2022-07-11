###############################################
# Standard gumpf
###############################################
cmake_minimum_required (VERSION 3.5)
project (SomaDSP)

add_library (SomaDSP SHARED util.c
                            frame_envelope_follower.c
                            biquad.c
                            so_lattice.c
                            dynamic_graphic_eq.c)
target_include_directories (SomaDSP PUBLIC ./)
target_link_libraries (SomaDSP PUBLIC m)

install (TARGETS SomaDSP
         LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
