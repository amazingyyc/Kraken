include(FetchContent)

FetchContent_Declare(
  eigen
  # GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_REPOSITORY https://github.com/eigenteam/eigen-git-mirror.git
  GIT_TAG        3.3.0
  SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/eigen
)

FetchContent_MakeAvailable(eigen)
