# CMakeList.txt : CMake project for CarDetectionAndCountingYOLOv5, include source and define
# project specific logic here.
#

# Add source to this project's executable.
add_executable (CarDetectionAndCountingYOLOv5 "CarDetectionAndCountingYOLOv5.cpp" "CarDetectionAndCountingYOLOv5.h"     "SORT.h" "SORT.cpp" "KalmanFilter.h" "KalmanFilter.cpp" "Blob.h" "Blob.cpp"  )

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET CarDetectionAndCountingYOLOv5 PROPERTY CXX_STANDARD 20)
endif()

# TODO: Add tests and install targets if needed.
set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/CarDetectionAndCountingYOLOv5/third_party/opencv/build)

find_package(OpenCV REQUIRED PATHS ${OpenCV_DIR})

include_directories(${OpenCV_INCLUDE_DIRS})

link_directories(${OpenCV_LIBRARY_DIRS})

target_link_libraries(CarDetectionAndCountingYOLOv5 ${OpenCV_LIBRARIES})

target_compile_definitions(CarDetectionAndCountingYOLOv5 PRIVATE 
    PROJECT_RELATIVE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/third_party/Yolo5cpp - Kopya/yolov5/yolov5"
)