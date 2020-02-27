
if(BUILD_MEX)
  add_subdirectory(matlab)
endif()

include_directories(${CMAKE_SOURCE_DIR}/src/core)

add_executable(test_dsplkgr test_dsplkgr.c)
target_link_libraries(test_dsplkgr PRIVATE egrss)
add_test(NAME core_dsplkgr COMMAND test_dsplkgr)

add_executable(test_dpotrf test_dpotrf.c)
target_link_libraries(test_dpotrf PRIVATE egrss)
add_test(NAME core_dpotrf COMMAND test_dpotrf)
