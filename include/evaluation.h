// Author: Vincenzo Sammartino

#ifndef CV_FINALPROJECT_EVALUATION_H
#define CV_FINALPROJECT_EVALUATION_H

#include <string>
#include <vector>
#include <opencv2/core.hpp>

#include "chessboard.h"
#include "io_functions.h"
#include "board_detector.h"
#include "chess_classifier.h"
#include "classifier_eval_config.h"

struct GroundTruthObject {
    ChessPieceCategory type = NONE;
    cv::Rect box;
};

struct EvaluationResult {
    // Base counts
    int totalGroundTruth = 0;
    int totalPredictions = 0;

    // IoU-based matching
    int matchedByIoU = 0;

    // Detection metric (we only print this)
    float detectionF1 = 0.0f;

    // Classification metrics on IoU matches
    int correctlyClassifiedOnMatches = 0;
    float classificationAccuracyOnMatches = 0.0f;
    float meanConfidenceOnMatches = 0.0f;
};

struct ImageRunOutput {
    std::vector<DetectionResult> results;
    EvaluationResult eval;
    bool hasEval = false;
};

struct DatasetEvaluationSummary {
    int numTestImages = 0;
    int loadedImages = 0;
    int failedLoads = 0;

    // per-image averages (normalized over loadedImages)
    double avgGtPerImage = 0.0;
    double avgPredPerImage = 0.0;
    double avgMatchesPerImage = 0.0;

    // per-image average percentages
    double meanDetectionF1Percent = 0.0; // mean(F1)*100
    double meanClsAccOnMatchesPercent = 0.0; // mean(acc_on_matches)*100
    double meanConfidenceOnMatchesPercent = 0.0; // mean(conf_on_matches)*100

    // useful to understand "how many matches are correct on average"
    double avgCorrectClsOnMatchesPerImage = 0.0;
};


/**
 * @brief Evaluation module: GT parsing (YOLO), IoU-based matching, and metrics.
 *
 * Note: by design choice, this class DOES NOT own BoardDetector/Classifier/Dataset1:
 * it receives them as references in its public methods (dependency injection).
 */
class Evaluation {
public:
    // =========================
    // Construction
    // =========================

    Evaluation() = default;

    // =========================
    // High-level API
    // =========================

    /**
     * @brief Evaluates a single image from the test set and saves the image with bboxes.
     */
    static bool runSingleImageEvaluation(const Dataset1 &dataset1,
                                         const BoardDetector &boardDetector,
                                         ChessClassifier &classifier,
                                         int testImageIndex,
                                         float iouThreshold = 0.30f);

    /**
     * @brief Evaluates all images in the test set, accumulating normalized metrics per image.
     */
    static DatasetEvaluationSummary runDatasetEvaluation(const Dataset1 &dataset1,
                                                         const BoardDetector &boardDetector,
                                                         ChessClassifier &classifier,
                                                         float iouThreshold = 0.30f);

    /**
     * @brief Prints the dataset evaluation summary to the console.
     */
    static void printDatasetEvaluationSummary(const DatasetEvaluationSummary &summary);

    // =========================
    // Core evaluation building blocks
    // =========================

    /**
     * @brief Loads ground-truth from YOLO labels (normalized xc, yc, w, h).
     */
    static std::vector<GroundTruthObject> loadGroundTruthFromYoloLabel(const std::string &labelPath,
                                                                       const cv::Size &imageSize);

    /**
     * @brief IoU between two bounding boxes.
     */
    static float computeIoU(const cv::Rect &a, const cv::Rect &b);

    /**
     * @brief Evaluates predictions vs GT using greedy IoU-based matching (sorted by confidence).
     * Returns the required metrics: base counts, matchedByIoU, F1, cls accuracy on matches, and mean conf on matches.
     */
    static EvaluationResult evaluateAgainstGroundTruth(const std::vector<DetectionResult> &predictions,
                                                       const std::vector<GroundTruthObject> &groundTruth,
                                                       float iouThreshold = 0.30f);

    /**
     * @brief Draws bboxes and labels (class + confidence) on the detections.
     */
    static cv::Mat drawClassifiedDetections(const cv::Mat &image,
                                            const std::vector<DetectionResult> &predictions);

private:
    // =========================
    // Internal helper (used by runSingle/runDataset)
    // =========================
    static ImageRunOutput runOnLoadedImage(const Dataset1 &dataset1,
                                           const BoardDetector &boardDetector,
                                           ChessClassifier &classifier,
                                           const cv::Mat &img,
                                           int index,
                                           float iouThreshold);

    static const EvaluationConfig config;
};

#endif // CV_FINALPROJECT_EVALUATION_H
