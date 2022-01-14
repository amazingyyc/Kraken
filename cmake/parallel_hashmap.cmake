include(FetchContent)

FetchContent_Declare(
  parallel_hashmap
  GIT_REPOSITORY https://github.com/greg7mdp/parallel-hashmap.git
  GIT_TAG        1.33
  SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/parallel_hashmap
)

FetchContent_MakeAvailable(parallel_hashmap)
