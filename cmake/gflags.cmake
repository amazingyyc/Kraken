include(FetchContent)

FetchContent_Declare(
  gflags
  GIT_REPOSITORY https://github.com/gflags/gflags
  GIT_TAG        v2.2.2
  SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/gflags
)

FetchContent_MakeAvailable(gflags)
