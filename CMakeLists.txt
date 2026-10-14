# Author: Leonardo Gusson, 2157199

cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

project(CV_FinalProject VERSION 1.0 LANGUAGES CXX)

find_package(OpenCV REQUIRED COMPONENTS
        core
        imgcodecs
        imgproc
        highgui
        xfeatures2d
        ml)

add_executable(${PROJECT_NAME}
        src/main.cpp
        src/io_functions.cpp
        src/chess_classifier.cpp
        src/evaluation.cpp
        src/board_detector.cpp
        src/geometry_utils.cpp
        src/chessboard.cpp
)

target_include_directories(${PROJECT_NAME} PRIVATE
        ${OpenCV_INCLUDE_DIRS}
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
)
target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS})

target_compile_definitions(${PROJECT_NAME} PRIVATE
        INPUT_FILES_DIR="${PROJECT_SOURCE_DIR}/input_files/"
        OUTPUT_FILES_DIR="${PROJECT_SOURCE_DIR}/output_files/"
        DATASET1_DIR="${PROJECT_SOURCE_DIR}/input_files/dataset1/"
        DATASET2_DIR="${PROJECT_SOURCE_DIR}/input_files/dataset2/"
)
