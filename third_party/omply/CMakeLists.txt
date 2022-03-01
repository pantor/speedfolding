cmake_minimum_required(VERSION 3.10)
project(omply)


set(CMAKE_CXX_STANDARD 17)


find_package(Eigen3 REQUIRED)
find_package(pybind11 REQUIRED)


include_directories(include include)


pybind11_add_module(_omply src/bindings.cpp src/trac_ik.cpp src/kdl_tl.cpp src/nlopt_ik.cpp)
target_link_libraries(_omply PRIVATE Eigen3::Eigen ompl orocos-kdl urdf kdl_parser fcl nlopt)
set_target_properties(_omply PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/omply")
