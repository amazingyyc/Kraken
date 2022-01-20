cmake_minimum_required(VERSION 3.6)

if(APPLE)
  set(CMAKE_C_COMPILER "gcc")
  set(CMAKE_CXX_COMPILER "g++")
endif(APPLE)

project(Kraken LANGUAGES C CXX)

# c use c11 c++ use 14 avoid zmq build error.
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

# try to find openmp
find_package(OpenMP)

if(OPENMP_FOUND)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")

  # add a OpenMP flag
  add_definitions(-DHAVE_OPENMP)
else()
  message(WARNING "Can not find OpenMP")
endif()

# get pytorch cmake path
execute_process(COMMAND python3 -c "import torch;print(torch.utils.cmake_prefix_path, end='')"
  OUTPUT_VARIABLE PYTORCH_CMAKE_PREFIX_PATH)
message(STATUS "PYTORCH_CMAKE_PREFIX_PATH:${PYTORCH_CMAKE_PREFIX_PATH}")

execute_process(COMMAND
  python3 -c "import torch.utils.cpp_extension;print(torch.utils.cpp_extension.library_paths()[0], end='')"
  OUTPUT_VARIABLE PYTORCH_LIBRARY_PREFIX_PATH)
message(STATUS "PYTORCH_LIBRARY_PREFIX_PATH:${PYTORCH_LIBRARY_PREFIX_PATH}")

# add fPIC flag
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
  message(STATUS "set CMAKE_BUILD_TYPE to Release.")
endif()

message(STATUS "CMAKE_BUILD_TYPE is: ${CMAKE_BUILD_TYPE}")

SET(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -Wall -g2 -ggdb")
SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -Wall")

# include dir
include_directories(kraken)

# use SYSTEM avoid warning
include_directories(SYSTEM third_party)

# source file
file(GLOB_RECURSE KRAKEN_HEAD_FILES "kraken/*.h")
file(GLOB_RECURSE KRAKEN_SRC_FILES "kraken/*.cc")

# test source file
file(GLOB_RECURSE KRAKEN_TEST_FILES "kraken/*_test.cc")

# for pybind11 API file
file(GLOB_RECURSE KRAKEN_PYBIND11_HEAD_FILES "kraken/pybind11/*.h")
file(GLOB_RECURSE KRAKEN_PYBIND11_SRC_FILES "kraken/pybind11/*.cc")

# remove test/main file
list(FILTER KRAKEN_SRC_FILES EXCLUDE REGEX ".*_test.cc")
list(FILTER KRAKEN_SRC_FILES EXCLUDE REGEX ".*_main.cc")

# remove for pybind11 file
list(REMOVE_ITEM KRAKEN_HEAD_FILES ${KRAKEN_PYBIND11_HEAD_FILES})
list(REMOVE_ITEM KRAKEN_SRC_FILES ${KRAKEN_PYBIND11_SRC_FILES})

# test file
file(GLOB_RECURSE KRAKEN_TEST_HEAD_FILES "kraken/test/*.h")
file(GLOB_RECURSE KRAKEN_TEST_SRC_FILES "kraken/test/*.cc")

list(FILTER KRAKEN_TEST_SRC_FILES EXCLUDE REGEX ".*_main.cc")

# #############################################################################################################
# include third_party cmake
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake;${CMAKE_MODULE_PATH}")

# include(googletest)
include(eigen)
include(pybind11)
include(gflags)
include(googletest)

set(SNAPPY_BUILD_TESTS OFF CACHE BOOL "Close snappy test build" FORCE)
set(SNAPPY_BUILD_BENCHMARKS OFF CACHE BOOL "Close snappy benchmark build" FORCE)
set(SNAPPY_INSTALL OFF CACHE BOOL "Close snappy install" FORCE)
include(snappy)

set(BUILD_TESTS OFF CACHE BOOL "set libzmq BUILD_TESTS to be OFF")
include(libzmq)

include(parallel_hashmap)

# ########################################################################################
# dynamic library
find_package(PythonLibs REQUIRED)
find_package(Torch REQUIRED PATHS "${PYTORCH_CMAKE_PREFIX_PATH}/Torch")

message(STATUS "TORCH_LIBRARIES:${TORCH_LIBRARIES}")
message(STATUS "KRAKEN_PYBIND11_SRC_FILES:${KRAKEN_PYBIND11_SRC_FILES}")

add_library(kraken_native SHARED
  ${KRAKEN_HEAD_FILES}
  ${KRAKEN_SRC_FILES}
  ${KRAKEN_PYBIND11_HEAD_FILES}
  ${KRAKEN_PYBIND11_SRC_FILES})

# snappy disable RTTI by default so we should disable the sub-class
set_source_files_properties(kraken/common/snappy.cc
  PROPERTIES
  COMPILE_FLAGS
  -fno-rtti)

# remove lib prefix for kraken_native
set_target_properties(kraken_native PROPERTIES PREFIX "")
target_include_directories(kraken_native PUBLIC ${PYTHON_INCLUDE_DIRS})
target_link_libraries(kraken_native
  libzmq-static
  snappy
  pybind11_headers
  ${PYTHON_LIBRARIES}
  ${TORCH_LIBRARIES}
  ${PYTORCH_LIBRARY_PREFIX_PATH}/libtorch_python.so)

# ########################################################################################
# kraken_server executable
add_executable(kraken_server
  kraken/executable/kraken_server_main.cc
  ${KRAKEN_HEAD_FILES}
  ${KRAKEN_SRC_FILES})
target_link_libraries(kraken_server libzmq-static snappy gflags)

# ########################################################################################
# kraken_test executable
add_executable(kraken_test
  kraken/test/kraken_test_main.cc
  ${KRAKEN_HEAD_FILES}
  ${KRAKEN_SRC_FILES}
  ${KRAKEN_TEST_HEAD_FILES}
  ${KRAKEN_TEST_SRC_FILES})
target_link_libraries(kraken_test libzmq-static snappy gtest)

# ########################################################################################
# run test
add_custom_target(test
  COMMAND ./kraken_test
  DEPENDS kraken_test)
