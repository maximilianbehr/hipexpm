# MIT License
#
# Copyright (c) 2024 Maximilian Behr
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

cmake_minimum_required(VERSION 3.23)
project(hipexpm LANGUAGES CXX HIP VERSION 2.0.0)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# hipexpm library
add_library(hipexpm SHARED hipexpm.hip)
target_include_directories(hipexpm PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(hipexpm PUBLIC hipblas hipsolver)
install(TARGETS hipexpm DESTINATION lib)
install(FILES hipexpm.h DESTINATION include)

# examples
foreach(x s d c z)
  add_executable(example_hipexpm${x} example_hipexpm${x}.hip)
  target_link_libraries(example_hipexpm${x} PUBLIC hipexpm)
  target_compile_options(example_hipexpm${x} PUBLIC -Wno-unused-result)
endforeach()
