cmake_minimum_required(VERSION 3.10)
project(phoxipy)


set(CMAKE_CXX_STANDARD 17)


# Why EGL?
find_package(OpenGL REQUIRED OpenGL)
find_package(PhoXi REQUIRED CONFIG PATHS "$ENV{PHOXI_CONTROL_PATH}")
find_package(pybind11 REQUIRED)


include_directories(include ${PHOXI_INCLUDE_DIRS})


pybind11_add_module(native_phoxi_driver src/native_driver.cpp)
target_link_libraries(native_phoxi_driver PRIVATE ${PHOXI_LIBRARY} rt OpenGL::GL OpenGL::GLU OSMesa)
set_target_properties(native_phoxi_driver PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/phoxi")
