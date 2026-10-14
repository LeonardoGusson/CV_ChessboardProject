// Author: Vincenzo Sammartino

#ifndef CV_FINALPROJECT_CLASSIFIER_EVAL_CONFIG_H
#define CV_FINALPROJECT_CLASSIFIER_EVAL_CONFIG_H

#include <string>

struct ChessClassifierConfig {
    // -----------------------------------------------------------------------------
    // Files
    // -----------------------------------------------------------------------------
    const std::string kModelPathRF = std::string(OUTPUT_FILES_DIR) + "chess_rf.yml";
    const std::string kVocabularyPath = std::string(OUTPUT_FILES_DIR) + "chess_vocabulary.yml";

    // -----------------------------------------------------------------------------
    // Hyper-parameters / thresholds
    // -----------------------------------------------------------------------------
    int kNumModelClasses = 6; // 6 chess categories only (no NONE in the model)
    int kSiftMaxFeaturesPerImage = 700; // SIFT cap per image
    int kPatchDescriptorBudget = 80; // max descriptors kept per ROI
    int kMinDescriptorsForPiece = 6; // descriptors required to classify a patch
    float kMinEdgeDensity = 0.010f; // quick "non-empty" gate (mask-normalized)
    float kTrainPositiveExpansion = 1.35f; // expand GT boxes to include context

    // K-means (global vocabulary)
    int kMaxDescriptorsForGlobalKMeans = 300000;
    float kGlobalKMeansSubsampleRatio = 0.75f;
    int kGlobalKMeansMaxIter = 300;
    double kGlobalKMeansEps = 1e-3;
    int kGlobalKMeansAttempts = 2;
};

struct EvaluationConfig {
    // Dataset split index used by Dataset1 (2 = test split in your project convention).
    int kDatasetTestSplit = 2;

    // Toggle extra outputs.
    bool kVerboseDetections = false; // print every predicted bbox + class
    bool kSave2DChessboard = true; // export chessboard_2d_*.png
};

#endif //CV_FINALPROJECT_CLASSIFIER_EVAL_CONFIG_H
