# CMakeLists.txt

# Older versions of CMake are likely to work just fine but, since
# I don't know where to cut off I just use the version I'm using
cmake_minimum_required(VERSION "3.17")

# name of this example project
project(maskrcnn-demo)

#set(CMAKE_OSX_ARCHITECTURES "x86_64")
list(APPEND CMAKE_PREFIX_PATH "/Users/youssefoumate/libtorch_Cpp/libtorch")
list(APPEND CMAKE_PREFIX_PATH "/Users/youssefoumate/libtorch_Cpp/vision/")


# set OpenCV_DIR variable equal to the path to the cmake
# files within the previously installed opencv program
set(OpenCV_DIR /Users/youssefoumate/opencv_cpp/build_opencv/lib/)

# Tell compiler to use C++ 14 features which is needed because
# Clang version is often behind in the XCode installation
set(CMAKE_CXX_STANDARD 17)

# configure the necessary common CMake environment variables
# needed to include and link the OpenCV program into this
# demo project, namely OpenCV_INCLUDE_DIRS and OpenCV_LIBS
find_package( OpenCV REQUIRED )

# tell the build to include the headers from OpenCV
include_directories( ${OpenCV_INCLUDE_DIRS} )

# specify the executable target to be built
add_executable(maskrcnn-demo ts_maskrcnn.cpp)

# tell it to link the executable target against OpenCV
target_link_libraries(maskrcnn-demo ${OpenCV_LIBS} )

# This due to LibTorch's version is the one included in the Python
# package that links to Python.
find_package(Python3 COMPONENTS Development)

# Add torchvision libraries

cmake_minimum_required(VERSION 3.10)
list(APPEND CMAKE_PREFIX_PATH "/Users/youssefoumate/libtorch_Cpp/vision/")

# The first thing do is to tell cmake to find the TorchVision library.
# The package pulls in all the necessary torch libraries,
# so there is no need to also add `find_package(Torch)` here.
find_package(TorchVision REQUIRED)

target_compile_features(maskrcnn-demo PUBLIC cxx_range_for)
target_link_libraries(maskrcnn-demo TorchVision::TorchVision)
set_property(TARGET maskrcnn-demo PROPERTY CXX_STANDARD 17)