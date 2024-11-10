# Set the minimum required CMake version
cmake_minimum_required(VERSION 3.28)

# Set the project name and specify C++ standard
project(linear_gp_project LANGUAGES CXX)

# Use C++20 standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS "-O3")

# Find the pybind11 package (Make sure pybind11 is installed)
find_package(pybind11 REQUIRED)

# Add the library sources
add_library(linear_gp SHARED
        python_bindings.cpp
        Program.cpp
        Mutator.cpp
)

# Link pybind11 to the linear_gp library
target_link_libraries(linear_gp PRIVATE pybind11::module)

# Set the output name to avoid prefixing with "lib"
set_target_properties(linear_gp PROPERTIES PREFIX "")

# Specify output directory for shared library
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY .)
