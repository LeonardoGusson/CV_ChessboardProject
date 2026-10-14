// Author: Leonardo Gusson, 2157199

#ifndef CV_FINALPROJECT_CHESS_CLASSIFIER_H
#define CV_FINALPROJECT_CHESS_CLASSIFIER_H

#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/ml.hpp>

#include "chessboard.h"     // ChessPieceCategory
#include "board_detector.h" // Detection
#include "io_functions.h"   // Dataset1, Dataset2
#include "classifier_eval_config.h"

struct DetectionResult {
    ChessPieceCategory type = NONE;
    cv::Rect refinedBox;
    float confidence = 0.0f;
};

class ChessClassifier {
public:
    ChessClassifier();

    // Train the full pipeline:
    // 1) extract SIFT descriptors (global), slice them by ROI/mask
    // 2) build BoW vocabulary with k-means
    // 3) build BoW train set
    // 4) train the final classifier (RTrees)
    void train(const Dataset1 &data1, const Dataset2 &data2);

    bool loadTrainedModel();

    bool saveTrainedModel() const;

    bool isModelEmpty() const { return model.empty(); }

    // Classify each detected ROI from BoardDetector output.
    // NOTE: the model is trained on 6 classes only (no NONE in the model).
    std::vector<DetectionResult> testImage(const Detection &detection);

private:
    // Optional preprocessing used both in training and inference.
    // Kept inside ChessClassifier because you removed preprocessing.{h,cpp}.
    static cv::Mat preprocessTrainingImage(const cv::Mat &img, int kernelSize = 5, double sigma = 1.0);

    // Bag-of-Words vocabulary creation (global k-means).
    void clusterCodebook(int k);

    // Build BoW histograms for each training sample + labels.
    void buildTrainDataset();

    // Train the final classifier (Random Forest).
    void trainModel();

    // Build a BoW histogram by matching descriptors to the vocabulary.
    cv::Mat computeHistogram(const cv::Mat &descriptors) const;

    // Confidence estimate from model votes.
    float computeClassificationConfidence(const cv::Mat &histogram) const;

    // Map between enum category and the model label space (0..5).
    static int categoryToModelLabel(ChessPieceCategory category);

    static ChessPieceCategory modelLabelToCategory(int label);

    // Slice global SIFT (computed once per image) inside ROI, optionally filtered by a local mask.
    cv::Mat addTrainingSampleFromGlobalSift(const cv::Mat &preprocessedImage,
                                            const std::vector<cv::KeyPoint> &globalKeypoints,
                                            const cv::Mat &globalDescriptors,
                                            const cv::Rect &roi,
                                            ChessPieceCategory label,
                                            const cv::Mat &localMask = cv::Mat());

    // -----------------------------------------------------------------------------
    // Helpers for RTrees votes -> per-class fractions (robust to OpenCV formats)
    // -----------------------------------------------------------------------------

    // Safely extract a single vote value from the RTrees votes matrix.
    static float voteValueAt(const cv::Mat &votes, int r, int c);

    // Convert RTrees raw votes into normalized per-class probabilities.
    static std::vector<float> extractClassVoteFractions(const cv::Mat &votes, int numClasses);

    // -----------------------------------------------------------------------------
    // Small utilities for ROIs
    // -----------------------------------------------------------------------------

    // Restrict a bounding box to fit strictly within the image dimensions.
    static cv::Rect clampRoi(const cv::Rect &roi, const cv::Size &size);

    // Expand a bounding box by a given scale factor, clamped to image bounds.
    static cv::Rect expandRoi(const cv::Rect &roi, const cv::Size &size, float scale);

    struct TrainImageMetadata {
        int startRow = 0;
        int numRows = 0;
        ChessPieceCategory type = NONE;
    };

    // Feature extractor
    cv::Ptr<cv::Feature2D> sift;

    // Trained model
    cv::Ptr<cv::ml::RTrees> model;
    cv::Mat vocabulary; // BoW centroids

    // Training buffers
    cv::Mat allDescriptors; // stacked descriptors for k-means
    std::vector<TrainImageMetadata> metadata;
    cv::Mat trainData; // BoW histograms
    cv::Mat trainLabels; // int labels 0..5

    static const ChessClassifierConfig config;
};

#endif // CV_FINALPROJECT_CHESS_CLASSIFIER_H
