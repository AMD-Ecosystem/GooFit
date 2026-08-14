target_sources(_goofit PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/GooPdf.cpp")
target_sources(_goofit PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/CombinePdf.cpp")
goofit_adopt_hip_sources(_goofit "${CMAKE_CURRENT_SOURCE_DIR}/GooPdf.cpp"
                         "${CMAKE_CURRENT_SOURCE_DIR}/CombinePdf.cpp")

add_subdirectory(basic)
add_subdirectory(combine)
if(GOOFIT_PHYSICS)
  add_subdirectory(physics)
endif()
add_subdirectory(utilities)
