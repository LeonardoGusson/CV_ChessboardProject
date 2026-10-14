// Author: Vincenzo Sammartino

#include "evaluation.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "board_detector.h"
#include "chess_classifier.h"
#include "io_functions.h"

const EvaluationConfig Evaluation::config;

// ============================================================
// Ground truth parsing helpers (YOLO labels -> pixel boxes)
// ============================================================

std::vector<GroundTruthObject> Evaluation::loadGroundTruthFromYoloLabel(const std::string &labelPath,
                                                                        const cv::Size &imageSize) {
    std::vector<GroundTruthObject> groundTruth;

    if (labelPath.empty() || imageSize.width <= 0 || imageSize.height <= 0) {
        return groundTruth;
    }

    std::ifstream labelFile(labelPath);
    if (!labelFile.is_open()) {
        return groundTruth;
    }

    // YOLO format: class xc yc w h (all normalized in [0,1])
    int classId = -1;
    float xc = 0.0f, yc = 0.0f, w = 0.0f, h = 0.0f;

    while (labelFile >> classId >> xc >> yc >> w >> h) {
        const ChessPieceCategory type = Dataset1::mapDataset1ClassToCategory(classId);
        if (type == NONE) {
            continue;
        }

        const int bw = static_cast<int>(w * static_cast<float>(imageSize.width));
        const int bh = static_cast<int>(h * static_cast<float>(imageSize.height));
        if (bw <= 0 || bh <= 0) {
            continue;
        }

        const int x = static_cast<int>((xc * static_cast<float>(imageSize.width)) - 0.5f * bw);
        const int y = static_cast<int>((yc * static_cast<float>(imageSize.height)) - 0.5f * bh);

        // Clamp to image bounds.
        const cv::Rect box = (cv::Rect(x, y, bw, bh) &
                              cv::Rect(0, 0, imageSize.width, imageSize.height));

        if (box.width <= 0 || box.height <= 0) {
            continue;
        }

        groundTruth.push_back({type, box});
    }

    return groundTruth;
}

// ============================================================
// Matching + metrics
// ============================================================

float Evaluation::computeIoU(const cv::Rect &a, const cv::Rect &b) {
    const cv::Rect inter = a & b;
    if (inter.width <= 0 || inter.height <= 0) {
        return 0.0f;
    }

    const float interArea = static_cast<float>(inter.area());
    const float unionArea = static_cast<float>(a.area() + b.area() - inter.area());
    if (unionArea <= 0.0f) {
        return 0.0f;
    }
    return interArea / unionArea;
}

EvaluationResult Evaluation::evaluateAgainstGroundTruth(const std::vector<DetectionResult> &predictions,
                                                        const std::vector<GroundTruthObject> &groundTruth,
                                                        const float iouThreshold) {
    EvaluationResult out{};
    out.totalGroundTruth = static_cast<int>(groundTruth.size());
    out.totalPredictions = static_cast<int>(predictions.size());

    if (out.totalGroundTruth == 0 || out.totalPredictions == 0) {
        // No GT or no predictions => no matches and all metrics are 0.
        return out;
    }

    // Greedy matching strategy:
    // 1) sort predictions by confidence (descending)
    // 2) for each prediction, match the best (highest IoU) unused GT
    // 3) accept match if IoU >= threshold
    std::vector<int> sortedIdx(predictions.size());
    std::iota(sortedIdx.begin(), sortedIdx.end(), 0);
    std::sort(sortedIdx.begin(), sortedIdx.end(),
              [&](const int a, const int b) { return predictions[a].confidence > predictions[b].confidence; });

    std::vector<bool> gtUsed(groundTruth.size(), false);
    float confSumOnMatches = 0.0f;
    int correctlyClassified = 0;

    for (const int predIdx: sortedIdx) {
        const DetectionResult &pred = predictions[predIdx];

        // A degenerate box can never be a valid match.
        if (pred.refinedBox.width <= 0 || pred.refinedBox.height <= 0) {
            continue;
        }

        float bestIoU = 0.0f;
        int bestGt = -1;

        for (int gtIdx = 0; gtIdx < static_cast<int>(groundTruth.size()); ++gtIdx) {
            if (gtUsed[gtIdx]) {
                continue;
            }
            const float iou = computeIoU(pred.refinedBox, groundTruth[gtIdx].box);
            if (iou > bestIoU) {
                bestIoU = iou;
                bestGt = gtIdx;
            }
        }

        if (bestGt >= 0 && bestIoU >= iouThreshold) {
            gtUsed[bestGt] = true;
            ++out.matchedByIoU;

            // Classification metrics computed only on matched detections.
            if (pred.type != NONE && pred.type == groundTruth[bestGt].type) {
                ++correctlyClassified;
            }
            confSumOnMatches += pred.confidence;
        }
    }

    // Detection F1 computed without explicitly storing FP:
    // precision = TP / totalPredictions
    // recall    = TP / totalGroundTruth
    // F1        = 2PR / (P + R)
    const float tp = static_cast<float>(out.matchedByIoU);
    const float precision = (out.totalPredictions > 0)
                                ? (tp / static_cast<float>(out.totalPredictions))
                                : 0.0f;
    const float recall = (out.totalGroundTruth > 0)
                             ? (tp / static_cast<float>(out.totalGroundTruth))
                             : 0.0f;

    out.detectionF1 = ((precision + recall) > 0.0f)
                          ? (2.0f * precision * recall) / (precision + recall)
                          : 0.0f;

    out.correctlyClassifiedOnMatches = correctlyClassified;
    if (out.matchedByIoU > 0) {
        out.classificationAccuracyOnMatches =
                static_cast<float>(correctlyClassified) / static_cast<float>(out.matchedByIoU);
        out.meanConfidenceOnMatches =
                confSumOnMatches / static_cast<float>(out.matchedByIoU);
    }

    return out;
}

// ============================================================
// Visualization helpers
// ============================================================

cv::Mat Evaluation::drawClassifiedDetections(const cv::Mat &image,
                                             const std::vector<DetectionResult> &predictions) {
    if (image.empty()) {
        return cv::Mat();
    }

    cv::Mat out;
    if (image.channels() == 1) {
        cv::cvtColor(image, out, cv::COLOR_GRAY2BGR);
    } else {
        out = image.clone();
    }

    auto typeToString = [](const ChessPieceCategory type) -> std::string {
        switch (type) {
            case BISHOP: return "bishop";
            case KING: return "king";
            case KNIGHT: return "knight";
            case PAWN: return "pawn";
            case QUEEN: return "queen";
            case ROOK: return "rook";
            default: return "empty";
        }
    };

    for (const auto &pred: predictions) {
        if (pred.refinedBox.width <= 0 || pred.refinedBox.height <= 0) {
            continue;
        }

        const cv::Scalar color = (pred.type == NONE)
                                     ? cv::Scalar(0, 0, 255) // red: unknown/none
                                     : cv::Scalar(0, 255, 0); // green: piece

        cv::rectangle(out, pred.refinedBox, color, 2);

        const std::string label =
                typeToString(pred.type) + " (" + cv::format("%.2f", pred.confidence) + ")";
        const int textY = std::max(0, pred.refinedBox.y - 6);

        cv::putText(out, label, cv::Point(pred.refinedBox.x, textY),
                    cv::FONT_HERSHEY_SIMPLEX, 0.45, color, 1);
    }

    return out;
}

// ============================================================
// High-level API (single image / dataset)
// ============================================================

ImageRunOutput Evaluation::runOnLoadedImage(const Dataset1 &dataset1,
                                            const BoardDetector &boardDetector,
                                            ChessClassifier &classifier,
                                            const cv::Mat &img,
                                            const int index,
                                            const float iouThreshold) {
    ImageRunOutput out;
    std::cout << "Processing image " << index << "..." << std::endl;

    // Board detection -> structured output (cells, geometry, etc.)
    const Detection detection = boardDetector.detect(img);

    // Classifier runs on the detector output and returns candidates.
    out.results = classifier.testImage(detection);

    if (config.kVerboseDetections) {
        std::cout << "Detections: " << out.results.size() << std::endl;
        for (size_t i = 0; i < out.results.size(); ++i) {
            const DetectionResult &r = out.results[i];
            std::cout << "[" << i << "] "
                    << "type=" << static_cast<int>(r.type)
                    << " box=(" << r.refinedBox.x << ","
                    << r.refinedBox.y << ","
                    << r.refinedBox.width << ","
                    << r.refinedBox.height << ")"
                    << " confidence=" << r.confidence
                    << std::endl;
        }
    }

    // If label exists, compute evaluation metrics.
    std::string labelPath;
    if (dataset1.getLabelPath(config.kDatasetTestSplit, index, labelPath)) {
        const std::vector<GroundTruthObject> gt = loadGroundTruthFromYoloLabel(labelPath, img.size());

        out.eval = evaluateAgainstGroundTruth(out.results, gt, iouThreshold);
        out.hasEval = true;

        // Print only the metrics we decided to keep.
        std::cout << "GT: " << out.eval.totalGroundTruth
                << " | Pred: " << out.eval.totalPredictions
                << " | Matches(IoU): " << out.eval.matchedByIoU
                << " | F1: " << out.eval.detectionF1
                << " | ClsCorrect: " << out.eval.correctlyClassifiedOnMatches
                << " | ClsAcc@Matches: " << out.eval.classificationAccuracyOnMatches
                << " | MeanConf@Matches: " << out.eval.meanConfidenceOnMatches
                << std::endl;
    }

    // Save visualization with predicted boxes + labels.
    const cv::Mat vis = drawClassifiedDetections(img, out.results);
    if (!vis.empty()) {
        const std::string outPath =
                std::string(OUTPUT_FILES_DIR) + "classified_test_" + std::to_string(index) + ".jpg";
        cv::imwrite(outPath, vis);
    }

    // export a 2D chessboard.
    if (config.kSave2DChessboard && detection.cellPositions.size() == 64) {
        std::vector<std::pair<int, ChessPieceCategory> > boardPieces;
        boardPieces.reserve(64);

        int occupiedIndex = 0;
        for (int i = 0; i < 64; ++i) {
            if (detection.cellPositions[i]) {
                if (occupiedIndex < static_cast<int>(out.results.size())) {
                    boardPieces.push_back({i, out.results[occupiedIndex].type});
                    ++occupiedIndex;
                } else {
                    boardPieces.push_back({i, NONE});
                }
            } else {
                boardPieces.push_back({i, NONE});
            }
        }

        ChessBoard chessBoard2D;
        const cv::Mat boardImage = chessBoard2D.drawChessboard(boardPieces, detection.isFirstWhite);
        if (!boardImage.empty()) {
            const std::string boardOutPath =
                    std::string(OUTPUT_FILES_DIR) + "chessboard_2d_" + std::to_string(index) + ".png";
            cv::imwrite(boardOutPath, boardImage);
        }
    }

    return out;
}

bool Evaluation::runSingleImageEvaluation(const Dataset1 &dataset1,
                                          const BoardDetector &boardDetector,
                                          ChessClassifier &classifier,
                                          const int testImageIndex,
                                          const float iouThreshold) {
    cv::Mat testImg;
    if (!dataset1.getImage(config.kDatasetTestSplit, testImageIndex, testImg) || testImg.empty()) {
        std::cerr << "Failed to load image index " << testImageIndex << "." << std::endl;
        return false;
    }

    runOnLoadedImage(dataset1, boardDetector, classifier, testImg, testImageIndex, iouThreshold);
    return true;
}

DatasetEvaluationSummary Evaluation::runDatasetEvaluation(const Dataset1 &dataset1,
                                                          const BoardDetector &boardDetector,
                                                          ChessClassifier &classifier,
                                                          const float iouThreshold) {
    DatasetEvaluationSummary summary;
    summary.numTestImages = static_cast<int>(dataset1.getAllTestImagesPaths().size());

    // We want dataset-level "generic" performance on large sets:
    // compute averages per-image (normalized by number of loaded images).
    double sumGt = 0.0;
    double sumPred = 0.0;
    double sumMatches = 0.0;

    double sumF1 = 0.0;
    double sumClsAcc = 0.0;
    double sumMeanConf = 0.0;
    double sumCorrectCls = 0.0;

    for (int i = 0; i < summary.numTestImages; ++i) {
        cv::Mat testImg;
        if (!dataset1.getImage(config.kDatasetTestSplit, i, testImg) || testImg.empty()) {
            ++summary.failedLoads;
            std::cerr << "Failed to load test image index " << i << "." << std::endl;
            continue;
        }

        ++summary.loadedImages;

        const ImageRunOutput imageOut =
                runOnLoadedImage(dataset1, boardDetector, classifier, testImg, i, iouThreshold);

        if (imageOut.hasEval) {
            sumGt += static_cast<double>(imageOut.eval.totalGroundTruth);
            sumPred += static_cast<double>(imageOut.eval.totalPredictions);
            sumMatches += static_cast<double>(imageOut.eval.matchedByIoU);

            sumF1 += static_cast<double>(imageOut.eval.detectionF1);
            sumClsAcc += static_cast<double>(imageOut.eval.classificationAccuracyOnMatches);
            sumMeanConf += static_cast<double>(imageOut.eval.meanConfidenceOnMatches);
            sumCorrectCls += static_cast<double>(imageOut.eval.correctlyClassifiedOnMatches);
        }
    }

    const int N = summary.loadedImages;
    if (N > 0) {
        summary.avgGtPerImage = sumGt / static_cast<double>(N);
        summary.avgPredPerImage = sumPred / static_cast<double>(N);
        summary.avgMatchesPerImage = sumMatches / static_cast<double>(N);
        summary.avgCorrectClsOnMatchesPerImage = sumCorrectCls / static_cast<double>(N);

        // Percent-like outputs for quick reading.
        summary.meanDetectionF1Percent = (sumF1 / static_cast<double>(N)) * 100.0;
        summary.meanClsAccOnMatchesPercent = (sumClsAcc / static_cast<double>(N)) * 100.0;
        summary.meanConfidenceOnMatchesPercent = (sumMeanConf / static_cast<double>(N)) * 100.0;
    }

    return summary;
}

void Evaluation::printDatasetEvaluationSummary(const DatasetEvaluationSummary &summary) {
    std::cout << "Directory test completed." << std::endl;
    std::cout << "Loaded images: " << summary.loadedImages << "/" << summary.numTestImages << std::endl;
    std::cout << "Failed loads: " << summary.failedLoads << std::endl;

    std::cout << "Avg GT objects / image: " << summary.avgGtPerImage << std::endl;
    std::cout << "Avg predictions / image: " << summary.avgPredPerImage << std::endl;
    std::cout << "Avg IoU matches / image: " << summary.avgMatchesPerImage << std::endl;

    std::cout << "Mean Detection F1 (%): " << summary.meanDetectionF1Percent << std::endl;
    std::cout << "Mean Classification Acc on matches (%): " << summary.meanClsAccOnMatchesPercent << std::endl;
    std::cout << "Mean Confidence on matches (%): " << summary.meanConfidenceOnMatchesPercent << std::endl;

    std::cout << "Avg correctly classified matches / image: "
            << summary.avgCorrectClsOnMatchesPerImage << std::endl;
}
