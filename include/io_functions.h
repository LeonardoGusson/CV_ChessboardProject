// Author: Leonardo Gusson, 2157199

#ifndef CV_FINALPROJECT_IO_FUNCTIONS_H
#define CV_FINALPROJECT_IO_FUNCTIONS_H

#include <map>
#include <vector>
#include <string>
#include "chessboard.h"

class Dataset1 {
public:
    Dataset1();

    /**
     * @brief Loads a specific image.
     * @param type Subset type (0: Train, 1: Val, 2: Test).
     * @param index Image index.
     * @param[out] img Destination matrix.
     * @return True on success, false otherwise.
     */
    bool getImage(int type, int index, cv::Mat &img) const;

    /**
     * @brief Gets the label file path for a given image.
     * @param type Subset type (0: Train, 1: Val, 2: Test).
     * @param imageIndex Image index.
     * @param[out] labelPath Destination string.
     * @return True on success, false otherwise.
     */
    bool getLabelPath(int type, int imageIndex, std::string &labelPath) const;

    /**
     * @brief Prompts the user via CLI for a valid test image index.
     * @return Selected valid index.
     */
    int askForTestImageIndex() const;

    /** @brief Returns all training image paths. */
    const std::vector<std::string> &getAllTrainImagesPaths() const { return trImagesNames; }

    /** @brief Returns all validation image paths. */
    const std::vector<std::string> &getAllValidImagesPaths() const { return valImagesNames; }

    /** @brief Returns all test image paths. */
    const std::vector<std::string> &getAllTestImagesPaths() const { return tsImagesNames; }

    /** @brief Maps the YOLO dataset class ID (0..11) to a ChessPieceCategory (BISHOP..ROOK). */
    static ChessPieceCategory mapDataset1ClassToCategory(int classId);

private:
    /**
     * @brief Validates if an index is within bounds.
     * @param type Subset type (0, 1, or 2).
     * @param index Index to check.
     * @return True if valid, false otherwise.
     */
    bool validateIndex(int type, int index) const;

    // Directory constants (assumed to be defined via CMake)
    const std::string trainImgsDir = DATASET1_DIR + std::string("train/images/");
    const std::string validImgsDir = DATASET1_DIR + std::string("valid/images/");
    const std::string testImgsDir = DATASET1_DIR + std::string("test/images/");
    const std::string trainLabelsDir = DATASET1_DIR + std::string("train/labels/");
    const std::string validLabelsDir = DATASET1_DIR + std::string("valid/labels/");
    const std::string testLabelsDir = DATASET1_DIR + std::string("test/labels/");

    // Vectors to store file paths
    std::vector<std::string> trImagesNames; /// Training image paths.
    std::vector<std::string> valImagesNames; /// Validation image paths.
    std::vector<std::string> tsImagesNames; /// Test image paths.
};


class Dataset2 {
public:
    Dataset2();

    /**
     * @brief Loads a specific image from a category.
     * @param cat Chess piece category.
     * @param index Image index.
     * @param[out] img Destination matrix.
     * @return True on success, false otherwise.
     */
    bool getImage(ChessPieceCategory cat, int index, cv::Mat &img) const;

    /**
     * @brief Gets the number of available images for a category.
     * @param cat Chess piece category.
     * @return Image count.
     */
    int getCategorySize(ChessPieceCategory cat) const;

    /**
     * @brief Gets the total number of images across all piece categories.
     * @return Total image count.
     */
    int getTrainingSetSize() const;

private:
    /**
     * @brief Validates if an index is within bounds for a category.
     * @param cat Chess piece category.
     * @param index Index to check.
     * @return True if valid, false otherwise.
     */
    bool validateIndex(ChessPieceCategory cat, int index) const;

    // Directory constants for each class
    const std::string trBishopDir = DATASET2_DIR + std::string("train/bishop/");
    const std::string trKingDir = DATASET2_DIR + std::string("train/king/");
    const std::string trKnightDir = DATASET2_DIR + std::string("train/knight/");
    const std::string trPawnDir = DATASET2_DIR + std::string("train/pawn/");
    const std::string trQueenDir = DATASET2_DIR + std::string("train/queen/");
    const std::string trRookDir = DATASET2_DIR + std::string("train/rook/");

    // Image paths keyed by category
    std::map<ChessPieceCategory, std::vector<std::string> > trImagesNames;
};

/**
 * @brief Extracts the filename without extension and parent directories.
 * @param path Full file path.
 * @return Extracted filename string.
 */
std::string getFileNameFromPath(const std::string &path);

/**
 * @brief Asks the user for a binary choice.
 * @param msg message.
 * @param optionTrue Character for the 'true' choice.
 * @param optionTrueDescr Description of the 'true' choice.
 * @param optionFalse Character for the 'false' choice.
 * @param optionFalseDesc Description of the 'false' choice.
 * @return True if optionTrue was selected, false if optionFalse.
 */
bool askForBinaryOptions(const std::string &msg, char optionTrue, const std::string &optionTrueDescr,
                         char optionFalse, const std::string &optionFalseDesc);

/**
 * @brief Saves an image to the pre-configured debug directory.
 * @param img Image matrix to save.
 * @param fileName Target filename without extension.
 */
void saveDebugImage(const cv::Mat &img, const std::string &fileName);

#endif //CV_FINALPROJECT_IO_FUNCTIONS_H
