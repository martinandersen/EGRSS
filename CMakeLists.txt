cmake_minimum_required (VERSION 3.10)
project (egrss VERSION 0.1.0)

option(BUILD_MEX "Build MEX files" OFF)
option(BUILD_DOC "Build documentation" OFF)
option(BUILD_EXAMPLES "Build examples" OFF)
option(TEST_PYTHON "Test Python package" OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    include(CTest)
endif()

add_subdirectory(src/core)
if(BUILD_MEX)
    set(MATLAB_ADDITIONAL_VERSIONS
            "R2019a=9.6"
            "R2019b=9.7"
            "R2020a=9.8"
            "R2020b=9.9"
            "R2021a=9.10"
            "R2021b=9.11"
            "R2022a=9.12"
            "R2022b=9.13"
            "R2023a=9.14"
            "R2023b=9.15"
    )
    add_subdirectory(src/matlab/mex)
    add_subdirectory(src/matlab/ref)
endif()
if(BUILD_DOC)
    add_subdirectory(doc)
endif()
if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME AND BUILD_TESTING)
    add_subdirectory(tests)
endif()
