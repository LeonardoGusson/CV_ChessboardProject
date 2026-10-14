// Author: Leonardo Gusson, 2157199

// constants defined in the CMakeLists.txt file
// input and output directory paths:
// INPUT_FILES_DIR, OUTPUT_FILES_DIR, DATASET1_DIR, DATASET2_DIR

#include <iostream>
#include <vector>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core.hpp>

#include "io_functions.h"

Dataset1::Dataset1() {
    cv::glob(trainImgsDir + "*.jpg", trImagesNames);
    cv::glob(validImgsDir + "*.jpg", valImagesNames);
    cv::glob(testImgsDir + "*.jpg", tsImagesNames);
}

bool Dataset1::getImage(const int type, const int index, cv::Mat &img) const {
    if (validateIndex(type, index)) {
        if (type == 0) {
            img = cv::imread(trImagesNames[index]);
            if (img.empty()) return false;
            return true;
        } else if (type == 1) {
            img = cv::imread(valImagesNames[index]);
            if (img.empty()) return false;
            return true;
        } else if (type == 2) {
            img = cv::imread(tsImagesNames[index]);
            if (img.empty()) return false;
            return true;
        }
    }
    return false;
}

bool Dataset1::getLabelPath(const int type, const int imageIndex, std::string &labelPath) const {
    if (validateIndex(type, imageIndex)) {
        if (type == 0) {
            labelPath = trainLabelsDir + getFileNameFromPath(trImagesNames[imageIndex]) + ".txt";
        } else if (type == 1) {
            labelPath = validLabelsDir + getFileNameFromPath(valImagesNames[imageIndex]) + ".txt";
        } else if (type == 2) {
            labelPath = testLabelsDir + getFileNameFromPath(tsImagesNames[imageIndex]) + ".txt";
        }
        return true;
    }
    return false;
}

int Dataset1::askForTestImageIndex() const {
    if (getAllTestImagesPaths().empty()) {
        std::cerr << "No test images available in dataset." << std::endl;
        return 0;
    }
    const int maxIndex = static_cast<int>(getAllTestImagesPaths().size()) - 1;

    while (true) {
        std::cout << "Enter image index (0 - " << maxIndex << "): ";
        std::string input;
        std::cin >> input;

        try {
            int index = std::stoi(input);
            if (index >= 0 && index <= maxIndex) {
                return index;
            }
            std::cout << "Index out of range! Try again.";
        } catch (...) {
            std::cout << "Invalid input (not a number). Try again.";
        }
    }
}

bool Dataset1::validateIndex(const int type, const int index) const {
    if (type < 0 || type > 2 || index < 0) return false;
    const int trSize = static_cast<int>(trImagesNames.size());
    const int valSize = static_cast<int>(valImagesNames.size());
    const int tsSize = static_cast<int>(tsImagesNames.size());
    if ((type == 0 && index >= trSize) ||
        (type == 1 && index >= valSize) ||
        (type == 2 && index >= tsSize))
        return false;
    return true;
}

ChessPieceCategory Dataset1::mapDataset1ClassToCategory(const int classId) {
    // Dataset label convention: 0..11; map black/white versions to the same category.
    if (classId < 0 || classId > 11) return NONE;

    const int mapped = classId % 6;
    if (mapped < BISHOP || mapped > ROOK) return NONE;

    return static_cast<ChessPieceCategory>(mapped);
}

Dataset2::Dataset2() {
    std::vector<std::string> tmp;

    cv::glob(trBishopDir + "*.jpg", trImagesNames[BISHOP]);
    cv::glob(trBishopDir + "*.png", tmp);
    trImagesNames[BISHOP].insert(trImagesNames[BISHOP].end(), tmp.begin(), tmp.end());

    tmp = std::vector<std::string>();
    cv::glob(trKingDir + "*.jpg", trImagesNames[KING]);
    cv::glob(trKingDir + "*.png", tmp);
    trImagesNames[KING].insert(trImagesNames[KING].end(), tmp.begin(), tmp.end());

    tmp = std::vector<std::string>();
    cv::glob(trKnightDir + "*.jpg", trImagesNames[KNIGHT]);
    cv::glob(trKnightDir + "*.png", tmp);
    trImagesNames[KNIGHT].insert(trImagesNames[KNIGHT].end(), tmp.begin(), tmp.end());

    tmp = std::vector<std::string>();
    cv::glob(trPawnDir + "*.jpg", trImagesNames[PAWN]);
    cv::glob(trPawnDir + "*.png", tmp);
    trImagesNames[PAWN].insert(trImagesNames[PAWN].end(), tmp.begin(), tmp.end());

    tmp = std::vector<std::string>();
    cv::glob(trQueenDir + "*.jpg", trImagesNames[QUEEN]);
    cv::glob(trQueenDir + "*.png", tmp);
    trImagesNames[QUEEN].insert(trImagesNames[QUEEN].end(), tmp.begin(), tmp.end());

    tmp = std::vector<std::string>();
    cv::glob(trRookDir + "*.jpg", trImagesNames[ROOK]);
    cv::glob(trRookDir + "*.png", tmp);
    trImagesNames[ROOK].insert(trImagesNames[ROOK].end(), tmp.begin(), tmp.end());
}

bool Dataset2::getImage(const ChessPieceCategory cat, const int index, cv::Mat &img) const {
    if (validateIndex(cat, index)) {
        img = cv::imread(trImagesNames.at(cat)[index]);
        if (img.empty()) return false;
        return true;
    }
    return false;
}

int Dataset2::getCategorySize(const ChessPieceCategory cat) const {
    if (cat == NONE) return 0;
    const auto it = trImagesNames.find(cat);
    if (it == trImagesNames.end()) return 0;
    return static_cast<int>(it->second.size());
}

int Dataset2::getTrainingSetSize() const {
    int count = 0;
    count += getCategorySize(ChessPieceCategory::PAWN);
    count += getCategorySize(ChessPieceCategory::KNIGHT);
    count += getCategorySize(ChessPieceCategory::QUEEN);
    count += getCategorySize(ChessPieceCategory::ROOK);
    count += getCategorySize(ChessPieceCategory::KING);
    count += getCategorySize(ChessPieceCategory::BISHOP);
    return count;
}

bool Dataset2::validateIndex(const ChessPieceCategory cat, const int index) const {
    if (cat == NONE || index < 0) return false;
    switch (cat) {
        case BISHOP:
            return index < static_cast<int>(trImagesNames.at(BISHOP).size());
        case KING:
            return index < static_cast<int>(trImagesNames.at(KING).size());
        case KNIGHT:
            return index < static_cast<int>(trImagesNames.at(KNIGHT).size());
        case PAWN:
            return index < static_cast<int>(trImagesNames.at(PAWN).size());
        case QUEEN:
            return index < static_cast<int>(trImagesNames.at(QUEEN).size());
        case ROOK:
            return index < static_cast<int>(trImagesNames.at(ROOK).size());
        default:
            return false;
    }
}

std::string getFileNameFromPath(const std::string &path) {
    const size_t slashPos = path.find_last_of("/\\");
    std::string filename = (slashPos != std::string::npos) ? path.substr(slashPos + 1) : path;
    const size_t dotPos = filename.find_last_of('.');
    return (dotPos != std::string::npos) ? filename.substr(0, dotPos) : filename;
}


bool askForBinaryOptions(const std::string &msg, const char optionTrue, const std::string &optionTrueDescr,
                         const char optionFalse,
                         const std::string &optionFalseDesc) {
    char input;
    bool type = false;
    bool valid = false;

    std::cout << msg << " Type " << optionTrue << " for " << optionTrueDescr << "; type " << optionFalse << " for " <<
            optionFalseDesc << std::endl;

    const char optTrue = static_cast<char>(std::toupper(static_cast<unsigned char>(optionTrue)));
    const char optFalse = static_cast<char>(std::toupper(static_cast<unsigned char>(optionFalse)));

    while (!valid) {
        std::cout << "Your choice (" << optTrue << "/" << optFalse << "):  ";
        std::cin >> input;

        input = std::toupper(static_cast<unsigned char>(input));
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        if (input == optTrue) {
            type = true;
            valid = true;
        } else if (input == optFalse) {
            type = false;
            valid = true;
        } else {
            std::cout << "Invalid input. Try again." << std::endl;
        }
    }

    return type;
}

void saveDebugImage(const cv::Mat &img, const std::string &fileName) {
    if (img.empty() || fileName.empty()) return;

    const std::string path = OUTPUT_FILES_DIR + std::string("debug_images/") + fileName + std::string(".jpg");
    try {
        cv::imwrite(path, img);
    } catch (const cv::Exception &ex) {
        std::cerr << "OpenCV exception on saving a debug image " << ex.what() << std::endl;
    }
}
