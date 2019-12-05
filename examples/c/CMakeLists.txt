find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

include_directories(${CMAKE_SOURCE_DIR}/src/core)
add_executable(ssregr ssregr.c)
target_link_libraries(ssregr PRIVATE egrss ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES})
target_link_options(ssregr PRIVATE ${LAPACK_LINKER_FLAGS} ${BLAS_LINKER_FLAGS})
