cmake_minimum_required(VERSION 3.6)

if(APPLE)
  set(CMAKE_C_COMPILER "gcc")
  set(CMAKE_CXX_COMPILER "g++")
endif(APPLE)

project(Kraken LANGUAGES C CXX)

# c use c11 c++ use 17 avoid zmq build error.
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# SIMD
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")

# try to find openmp
find_package(OpenMP)

if(OPENMP_FOUND)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS
      "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")

  # add a OpenMP flag
  add_definitions(-DHAVE_OPENMP)
else()
  message(WARNING "Can not find OpenMP")
endif()

# get pytorch cmake path
execute_process(
  COMMAND python3 -c "import torch;print(torch.utils.cmake_prefix_path, end='')"
  OUTPUT_VARIABLE PYTORCH_CMAKE_PREFIX_PATH)
message(STATUS "PYTORCH_CMAKE_PREFIX_PATH:${PYTORCH_CMAKE_PREFIX_PATH}")

execute_process(
  COMMAND
    python3 -c
    "import torch.utils.cpp_extension;print(torch.utils.cpp_extension.library_paths()[0], end='')"
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

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -Wall -g2 -ggdb")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -Wall")

# include dir
include_directories(kraken)

# use SYSTEM avoid warning
include_directories(SYSTEM third_party)
include_directories(SYSTEM third_party/configor/include)

# source file
file(GLOB_RECURSE KRAKEN_HEAD_FILES "kraken/*.h")
file(GLOB_RECURSE KRAKEN_SRC_FILES "kraken/*.cc")

# test source file
file(GLOB_RECURSE KRAKEN_TEST_FILES "kraken/*_test.cc")

# for pytorch API file
file(GLOB_RECURSE KRAKEN_PYTORCH_HEAD_FILES "kraken/pytorch/*.h")
file(GLOB_RECURSE KRAKEN_PYTORCH_SRC_FILES "kraken/pytorch/*.cc")

# watcher file
file(GLOB_RECURSE KRAKEN_WATCHER_HEAD_FILES "kraken/watch/*.h")
file(GLOB_RECURSE KRAKEN_WATCHER_SRC_FILES "kraken/watch/*.cc")

# remove test/main file
list(FILTER KRAKEN_SRC_FILES EXCLUDE REGEX ".*_test.cc")
list(FILTER KRAKEN_SRC_FILES EXCLUDE REGEX ".*_main.cc")

# remove for pytorch file
list(REMOVE_ITEM KRAKEN_HEAD_FILES ${KRAKEN_PYTORCH_HEAD_FILES})
list(REMOVE_ITEM KRAKEN_SRC_FILES ${KRAKEN_PYTORCH_SRC_FILES})

# remove watcher file
list(REMOVE_ITEM KRAKEN_HEAD_FILES ${KRAKEN_WATCHER_HEAD_FILES})
list(REMOVE_ITEM KRAKEN_SRC_FILES ${KRAKEN_WATCHER_SRC_FILES})

# test file
file(GLOB_RECURSE KRAKEN_TEST_HEAD_FILES "kraken/test/*.h")
file(GLOB_RECURSE KRAKEN_TEST_SRC_FILES "kraken/test/*.cc")

list(FILTER KRAKEN_TEST_SRC_FILES EXCLUDE REGEX ".*_main.cc")

# ##############################################################################
# include third_party cmake
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake;${CMAKE_MODULE_PATH}")

# setup eigen build
add_definitions(-DEIGEN_USE_THREADS -DEIGEN_FAST_MATH -DEIGEN_NO_CUDA)

include(eigen)
include(pybind11)
include(gflags)
include(googletest)
include(configor)
include(libcuckoo)

set(SNAPPY_BUILD_TESTS
    OFF
    CACHE BOOL "Close snappy test build" FORCE)
set(SNAPPY_BUILD_BENCHMARKS
    OFF
    CACHE BOOL "Close snappy benchmark build" FORCE)
set(SNAPPY_INSTALL
    OFF
    CACHE BOOL "Close snappy install" FORCE)
include(snappy)

set(BUILD_TESTS
    OFF
    CACHE BOOL "set libzmq BUILD_TESTS to be OFF")
include(libzmq)

# snappy disable RTTI by default so we should disable the sub-class
set_source_files_properties(kraken/common/snappy.cc PROPERTIES COMPILE_FLAGS
                                                               -fno-rtti)

# ##############################################################################
# dynamic library
find_package(PythonLibs REQUIRED)
message(STATUS "PYTHON_LIBRARIES:${PYTHON_LIBRARIES}")

find_package(Torch REQUIRED PATHS "${PYTORCH_CMAKE_PREFIX_PATH}/Torch")

message(STATUS "TORCH_LIBRARIES:${TORCH_LIBRARIES}")
message(STATUS "KRAKEN_PYTORCH_SRC_FILES:${KRAKEN_PYTORCH_SRC_FILES}")

# ##############################################################################
# kraken_native
add_library(
  kraken_native SHARED ${KRAKEN_HEAD_FILES} ${KRAKEN_SRC_FILES}
                       ${KRAKEN_PYTORCH_HEAD_FILES} ${KRAKEN_PYTORCH_SRC_FILES})

# remove lib prefix for kraken_native
set_target_properties(kraken_native PROPERTIES PREFIX "")
target_include_directories(kraken_native PUBLIC ${PYTHON_INCLUDE_DIRS})
target_link_libraries(
  kraken_native
  stdc++fs
  libzmq-static
  snappy
  libcuckoo
  pybind11_headers
  ${PYTHON_LIBRARIES}
  ${TORCH_LIBRARIES}
  ${PYTORCH_LIBRARY_PREFIX_PATH}/libtorch_python.so)

# ##############################################################################
# kraken watcher add_library( kraken_watcher SHARED ${KRAKEN_HEAD_FILES}
# ${KRAKEN_SRC_FILES} ${KRAKEN_WATCHER_HEAD_FILES} ${KRAKEN_WATCHER_SRC_FILES})

# # remove lib prefix for kraken_watcher set_target_properties(kraken_watcher
# PROPERTIES PREFIX "") target_include_directories(kraken_watcher PUBLIC
# ${PYTHON_INCLUDE_DIRS}) target_link_libraries( kraken_watcher stdc++fs
# libzmq-static snappy libcuckoo pybind11_headers ${PYTHON_LIBRARIES}
# ${TORCH_LIBRARIES} ${PYTORCH_LIBRARY_PREFIX_PATH}/libtorch_python.so)

# ##############################################################################
# ps_server executable
add_executable(ps_server kraken/executable/ps_server_main.cc
                         ${KRAKEN_HEAD_FILES} ${KRAKEN_SRC_FILES})
target_link_libraries(ps_server stdc++fs libzmq-static snappy libcuckoo gflags)

# ##############################################################################
# ps_server executable
add_executable(scheduler_server kraken/executable/scheduler_server_main.cc
                                ${KRAKEN_HEAD_FILES} ${KRAKEN_SRC_FILES})
target_link_libraries(scheduler_server stdc++fs libzmq-static snappy libcuckoo
                      gflags)

# ##############################################################################
# kraken_test executable
add_executable(
  kraken_test
  kraken/test/kraken_test_main.cc ${KRAKEN_HEAD_FILES} ${KRAKEN_SRC_FILES}
  ${KRAKEN_TEST_HEAD_FILES} ${KRAKEN_TEST_SRC_FILES})
target_link_libraries(kraken_test stdc++fs libzmq-static snappy libcuckoo gtest)

# ##############################################################################
# run test
add_custom_target(
  test
  COMMAND ./kraken_test
  DEPENDS kraken_test)
