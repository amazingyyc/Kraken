include(FetchContent)

FetchContent_Declare(
  libcuckoo
  GIT_REPOSITORY https://github.com/efficient/libcuckoo.git
  GIT_TAG        v0.3
  SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/libcuckoo
)

FetchContent_MakeAvailable(libcuckoo)
