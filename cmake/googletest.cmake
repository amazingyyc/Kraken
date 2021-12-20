include(FetchContent)

FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        release-1.11.0
  SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/googletest
)

FetchContent_MakeAvailable(googletest)
