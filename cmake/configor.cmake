include(FetchContent)

FetchContent_Declare(
  configor
  GIT_REPOSITORY https://github.com/Nomango/configor.git
  GIT_TAG        v0.9.14
  SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/configor
)

FetchContent_MakeAvailable(configor)
