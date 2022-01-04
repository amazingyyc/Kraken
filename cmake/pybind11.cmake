include(FetchContent)

FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11
  GIT_TAG        v2.8.1
  SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/pybind11
)

FetchContent_MakeAvailable(pybind11)
