cmake_minimum_required(VERSION 3.6)

if(APPLE)
  set(CMAKE_C_COMPILER "gcc")
  set(CMAKE_CXX_COMPILER "g++")
endif(APPLE)

project(kGraph LANGUAGES C CXX)

# c use c11 c++ use 14 avoid zmq build error.
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

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
include_directories(third_party)

# source file
file(GLOB_RECURSE KRAKEN_HEAD_FILES "kraken/*.h")
file(GLOB_RECURSE KRAKEN_SRC_FILES "kraken/*.cc")

# test source file
file(GLOB_RECURSE KRAKEN_TEST_FILES "kgraph/*_test.cc")

# remove test/main file
list(FILTER KRAKEN_SRC_FILES EXCLUDE REGEX ".*_test.cc")
list(FILTER KRAKEN_SRC_FILES EXCLUDE REGEX ".*_main.cc")

##############################################################################################################
# include third_party cmake
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake;${CMAKE_MODULE_PATH}")

# include(googletest)
include(eigen)

set(BUILD_TESTS OFF CACHE BOOL "set libzmq BUILD_TESTS to be OFF")
include(libzmq)

#########################################################################################
# create library
add_library(kraken ${KRAKEN_HEAD_FILES} ${KRAKEN_SRC_FILES})
target_link_libraries(kraken libzmq-static)

#########################################################################################
# server_test executable
add_executable(server_test kraken/executable/server_test_main.cc)
target_link_libraries(server_test kraken)

# client test executable
add_executable(caller_test kraken/executable/caller_test_main.cc)
target_link_libraries(caller_test kraken)
