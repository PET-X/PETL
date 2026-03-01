# cmake version 3.23 or higher is needed to support 
# the argument CUDA_ARCHITECTURES all-major
# You could use cmake as old as version 3.18 if you replaced
# the line below that specifies all-major with the line below
# it which specifies the list of CUDA architectures 
#cmake_minimum_required(VERSION 3.23 FATAL_ERROR)
cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(petl CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDA 11.7 REQUIRED)
find_package(OpenMP REQUIRED)
#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -O3)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -D__INCLUDE_CUFFT)
add_compile_options(-D__INCLUDE_CUFFT)
add_compile_options(-O3)

if (WIN32)
  set(CMAKE_SHARED_LIBRARY_PREFIX "lib")
  set(CMAKE_STATIC_LIBRARY_PREFIX "lib")
endif()

#set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/..)
#set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/..)
#set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/..)

set(HEADER
  allocate.h
  analytic_ray_tracing.h
  analytic_ray_tracing_gpu.cuh
  binning.h
  cuda_utils.cuh
  data_cube.h
  list_mode_queue.h
  list_of_models.h
  parameters.h
  petl_c_interface.h
  petl_defines.h
  petl.h
  pfdr.cuh
  phantom.h
  planogram.h
  projectors.cuh
  ramp_filter.cuh
  total_variation.cuh
  median_filter.cuh
  vector_ops.h
  relative_differences.cuh
)

set(SRC_CPP
  allocate.cpp
  analytic_ray_tracing.cpp
  binning.cpp
  data_cube.cpp
  list_mode_queue.cpp
  list_of_models.cpp
  parameters.cpp
  petl_c_interface.cpp
  petl.cpp
  phantom.cpp
  planogram.cpp
  vector_ops.cpp
)

set(SRC_CU
  analytic_ray_tracing_gpu.cu
  cuda_utils.cu
  pfdr.cu
  projectors.cu
  ramp_filter.cu
  total_variation.cu
  median_filter.cu
  relative_differences.cu
)

include_directories(
  ./
  ${CUDA_INCLUDE_DIRS}
)

#add_library(petl SHARED main_projector_ctype.cpp parameters.cpp projectors_cpu.cpp projectors.cu projectors_SF.cu)
#target_link_libraries(petl "${TORCH_LIBRARIES}")
add_library(petl SHARED
  ${HEADER}
  ${SRC_CPP}
  ${SRC_CU}
)

# <<<--- Create filters for subfolders for Visual Studio
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

file(GLOB_RECURSE _source_list
  LIST_DIRECTORIES false
  ${HEADER}
  ${SRC_CPP}
  ${SRC_CU}
)

foreach(_source IN ITEMS ${_source_list})
  get_filename_component(_source_path "${_source}" PATH)
  file(RELATIVE_PATH _source_path_rel "${PROJECT_SOURCE_DIR}" "${_source_path}")
  string(REPLACE "/" "\\" _group_path "${_source_path_rel}")
  source_group("${_group_path}" FILES "${_source}")
endforeach()
# --->>>

target_link_libraries(${PROJECT_NAME}
  OpenMP::OpenMP_CXX
  ${CUDA_LIBRARIES}
  ${CUDA_cublas_LIBRARY}
  ${CUDA_cufft_LIBRARY}
)

target_compile_options(petl PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
  --use_fast_math
>)

#set_property(TARGET petl PROPERTY CXX_STANDARD 14)
set_property(TARGET petl PROPERTY CUDA_ARCHITECTURES native)
#if(CMAKE_VERSION VERSION_GREATER "3.23")
#    message("Building for all major cuda architectures")
#    set_property(TARGET petl PROPERTY CUDA_ARCHITECTURES all-major)
#endif()
#set_property(TARGET petl PROPERTY CUDA_ARCHITECTURES all-major)
#set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_ARCHITECTURES 60 61 62 70 72 75 80 86 87 89)
#set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_ARCHITECTURES 75 86 87 89)
#set_property(TARGET petl PROPERTY CUDA_ARCHITECTURES OFF)

install(TARGETS ${PROJECT_NAME}
  PUBLIC_HEADER DESTINATION include
  RUNTIME DESTINATION bin
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
)
