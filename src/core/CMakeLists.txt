
FILE(GLOB CORE *.c)

add_library (egrss ${CORE})
if(NOT MSVC)
  target_link_libraries(egrss m)
  set_property(TARGET egrss PROPERTY POSITION_INDEPENDENT_CODE ON)
endif()
if(BUILD_SHARED_LIBS)
  message(Building shared library)
  set_property(TARGET egrss PROPERTY VERSION ${PROJECT_VERSION})
  set_property(TARGET egrss PROPERTY SOVERSION ${PROJECT_VERSION})
endif()

if(MSVC)
  if(MSVC_VERSION GREATER_EQUAL 1900)
    # MSVC 14+ (1900) implements restricted pointers via "__restrict" instead of "restrict"
    target_compile_definitions(egrss PUBLIC restrict=__restrict)
  else()
    # Earlier versions do not support restricted pointers
    target_compile_definitions(egrss PUBLIC restrict=)
  endif()
endif()
