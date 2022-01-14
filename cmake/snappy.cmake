include(FetchContent)

FetchContent_Declare(
  snappy
  GIT_REPOSITORY https://github.com/google/snappy.git
  GIT_TAG        1.1.9
  SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/snappy
)

FetchContent_MakeAvailable(snappy)
