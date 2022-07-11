###############################################
# Standard gumpf
###############################################
cmake_minimum_required (VERSION 3.5)
project (SexyZebra)

set (BUNDLE_DIR SexyZebra.lv2)

###############################################
# That's a Sexy Zebra!
###############################################
add_library (SexyZebra SHARED SexyZebra.c)
target_compile_options (SexyZebra PRIVATE -fvisibility=hidden)
set_target_properties (SexyZebra PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${BUNDLE_DIR})
target_include_directories (SexyZebra PRIVATE ../generate_ttl)
target_link_libraries (SexyZebra PRIVATE SomaDSP)

###############################################
# ttl generation
###############################################
add_executable (generate_ttl ../generate_ttl/generate_ttl.c
	                     SexyZebra.c)
target_include_directories (generate_ttl PRIVATE ../generate_ttl)
target_link_libraries (generate_ttl PRIVATE serd-0 SomaDSP)

###############################################
# ttl files
###############################################
set (TTL_FILES ${BUNDLE_DIR}/manifest.ttl
               ${BUNDLE_DIR}/SexyZebra10.ttl
               ${BUNDLE_DIR}/SexyZebra30.ttl)

add_custom_command (OUTPUT ${TTL_FILES}
                    COMMAND generate_ttl $<TARGET_FILE_NAME:SexyZebra>
                    COMMAND gawk -i inplace -f ../../../generate_ttl/remove_quotes.awk SexyZebra10.ttl
                    COMMAND gawk -i inplace -f ../../../generate_ttl/remove_quotes.awk SexyZebra30.ttl
                    DEPENDS generate_ttl
                    WORKING_DIRECTORY ${BUNDLE_DIR})
add_custom_target (ttls ALL DEPENDS ${TTL_FILES})

###############################################
# install bundle
###############################################
if (DEFINED ENV{LV2_PATH})
	set (INSTALL_DIR $ENV{LV2_PATH}/${BUNDLE_DIR})
else()
	set (INSTALL_DIR /usr/local/lib/lv2/${BUNDLE_DIR})
endif()

install (TARGETS SexyZebra
         DESTINATION ${INSTALL_DIR})

install (FILES ${PROJECT_BINARY_DIR}/${BUNDLE_DIR}/manifest.ttl
               ${PROJECT_BINARY_DIR}/${BUNDLE_DIR}/SexyZebra10.ttl
               ${PROJECT_BINARY_DIR}/${BUNDLE_DIR}/SexyZebra30.ttl
         DESTINATION ${INSTALL_DIR})
