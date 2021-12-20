include(FetchContent)

FetchContent_Declare(
  eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG        3.4
  SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/eigen
)

FetchContent_MakeAvailable(eigen)
