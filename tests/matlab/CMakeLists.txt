find_package(Matlab COMPONENTS MAIN_PROGRAM REQUIRED)

# Test reference implementation
matlab_add_unit_test(
  NAME matlab_ref
  UNITTEST_FILE ${CMAKE_CURRENT_SOURCE_DIR}/egrss_test.m
  ADDITIONAL_PATH ${CMAKE_BINARY_DIR}/src/matlab/ref
)

# Test MEX implementation
matlab_add_unit_test(
  NAME matlab_mex
  UNITTEST_FILE ${CMAKE_CURRENT_SOURCE_DIR}/egrss_test.m
  ADDITIONAL_PATH ${CMAKE_BINARY_DIR}/src/matlab/mex
)
