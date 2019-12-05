
include_directories(${CMAKE_SOURCE_DIR}/src/core)
find_package(Matlab COMPONENTS MX_LIBRARY REQUIRED)

file(GLOB mfiles ../common/*.m)
add_custom_target(egrss_matlab_common ALL
                   COMMAND ${CMAKE_COMMAND} -E copy ${mfiles} ${CMAKE_CURRENT_BINARY_DIR})

matlab_add_mex(
  NAME egrss_generators
  SRC generators.c
  DOCUMENTATION ${CMAKE_CURRENT_SOURCE_DIR}/generators.txt
  LINK_TO egrss
)
matlab_add_mex(
  NAME egrss_gemv
  SRC gemv.c
  DOCUMENTATION ${CMAKE_CURRENT_SOURCE_DIR}/gemv.txt
  LINK_TO egrss
)
matlab_add_mex(
  NAME egrss_symv
  SRC symv.c
  DOCUMENTATION ${CMAKE_CURRENT_SOURCE_DIR}/symv.txt
  LINK_TO egrss
)
matlab_add_mex(
  NAME egrss_potrf
  SRC potrf.c
  DOCUMENTATION ${CMAKE_CURRENT_SOURCE_DIR}/potrf.txt
  LINK_TO egrss
)
matlab_add_mex(
  NAME egrss_trmv
  SRC trmv.c
  DOCUMENTATION ${CMAKE_CURRENT_SOURCE_DIR}/trmv.txt
  LINK_TO egrss
)
matlab_add_mex(
  NAME egrss_trsv
  SRC trsv.c
  DOCUMENTATION ${CMAKE_CURRENT_SOURCE_DIR}/trsv.txt
  LINK_TO egrss
)
matlab_add_mex(
  NAME egrss_trnrms
  SRC trnrms.c
  DOCUMENTATION ${CMAKE_CURRENT_SOURCE_DIR}/trnrms.txt
  LINK_TO egrss
)
