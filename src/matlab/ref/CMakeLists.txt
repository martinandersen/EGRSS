file(GLOB mfiles *.m ../common/*.m)
add_custom_target(egrss_matlab_reference ALL
                   COMMAND ${CMAKE_COMMAND} -E copy ${mfiles} ${CMAKE_CURRENT_BINARY_DIR})
