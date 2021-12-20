include(FetchContent)

FetchContent_Declare(
  libzmq
  GIT_REPOSITORY https://github.com/zeromq/libzmq.git
  GIT_TAG        v4.3.4
  SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/libzmq
)

FetchContent_MakeAvailable(libzmq)
