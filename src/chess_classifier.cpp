// Author: Vincenzo Sammartino

#include "chess_classifier.h"

#include <opencv2/core/persistence.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <numeric>

const ChessClassifierConfig ChessClassifier::config;

// =============================================================================
// Label mapping (model has ONLY 6 categories, no NONE)
// =============================================================================
int ChessClassifier::categoryToModelLabel(ChessPieceCategory category) {
    // The model does not include NONE. If it happens, it's a training logic error.
    if (category == NONE) return -1;
    return static_cast<int>(category); // expected 0..5
}

ChessPieceCategory ChessClassifier::modelLabelToCategory(int label) {
    if (label >= static_cast<int>(BISHOP) && label <= static_cast<int>(ROOK)) {
        return static_cast<ChessPieceCategory>(label); // 0..5
    }
    return NONE;
}

// =============================================================================
// Ctor / model I/O
// =============================================================================
ChessClassifier::ChessClassifier() {
    sift = cv::SIFT::create(
        config.kSiftMaxFeaturesPerImage, // nFeatures
        6, // nOctaveLayers
        0.015f, // contrastThreshold
        15.0f, // edgeThreshold
        1.0f, // sigma
        CV_32F
    );
}

bool ChessClassifier::loadTrainedModel() {
    // Load vocabulary (BoW codebook)
    cv::FileStorage fs(config.kVocabularyPath, cv::FileStorage::READ);
    if (!fs.isOpened()) return false;

    cv::Mat loadedVocabulary;
    fs["vocabulary"] >> loadedVocabulary;
    fs.release();
    if (loadedVocabulary.empty()) return false;

    // Load Random Forest
    cv::Ptr<cv::ml::RTrees> loadedModel;
    try {
        loadedModel = cv::Algorithm::load<cv::ml::RTrees>(config.kModelPathRF);
    } catch (const cv::Exception &) {
        return false;
    }
    if (loadedModel.empty()) return false;

    vocabulary = loadedVocabulary;
    model = loadedModel;
    return true;
}

bool ChessClassifier::saveTrainedModel() const {
    if (model.empty() || vocabulary.empty()) return false;

    try {
        model->save(config.kModelPathRF);
    } catch (const cv::Exception &) {
        return false;
    }

    cv::FileStorage fs(config.kVocabularyPath, cv::FileStorage::WRITE);
    if (!fs.isOpened()) return false;
    fs << "vocabulary" << vocabulary;
    fs.release();
    return true;
}

// =============================================================================
// Training
// =============================================================================
void ChessClassifier::train(const Dataset1 &data1, const Dataset2 &data2) {
    // Reset internal training buffers
    allDescriptors.release();
    metadata.clear();
    vocabulary.release();
    trainData.release();
    trainLabels.release();
    model.release();

    // -----------------------------
    // DATASET 1: images with labels
    // -----------------------------
    const auto &trainPaths = data1.getAllTrainImagesPaths();
    const int numTrainImages = static_cast<int>(trainPaths.size());

    std::cout << "Training on dataset 1:" << std::endl;
    for (int i = 0; i < numTrainImages; ++i) {
        std::cout << "\r>>> Processing image " << i << " / " << numTrainImages << std::flush;

        cv::Mat image;
        if (!data1.getImage(0, i, image) || image.empty()) continue;

        const cv::Mat pre = preprocessTrainingImage(image);
        if (pre.empty()) continue;

        // Compute global SIFT once, then slice descriptors per ROI
        std::vector<cv::KeyPoint> globalKeypoints;
        cv::Mat globalDescriptors;
        sift->detectAndCompute(pre, cv::noArray(), globalKeypoints, globalDescriptors);
        if (globalDescriptors.empty() || globalKeypoints.empty()) continue;

        std::string labelPath;
        if (!data1.getLabelPath(0, i, labelPath)) continue;

        std::ifstream labelFile(labelPath);
        if (!labelFile.is_open()) continue;

        int classId = -1;
        float xc = 0.f, yc = 0.f, w = 0.f, h = 0.f;

        while (labelFile >> classId >> xc >> yc >> w >> h) {
            const ChessPieceCategory cat = Dataset1::mapDataset1ClassToCategory(classId);
            if (cat == NONE) continue;

            const int imgW = image.cols;
            const int imgH = image.rows;
            const int bw = static_cast<int>(w * imgW);
            const int bh = static_cast<int>(h * imgH);
            if (bw <= 1 || bh <= 1) continue;

            const int x = static_cast<int>(xc * imgW - 0.5f * bw);
            const int y = static_cast<int>(yc * imgH - 0.5f * bh);

            const cv::Rect roi = clampRoi(cv::Rect(x, y, bw, bh), image.size());
            if (roi.width <= 1 || roi.height <= 1) continue;

            const cv::Rect expanded = expandRoi(roi, image.size(), config.kTrainPositiveExpansion);
            if (expanded.width <= 1 || expanded.height <= 1) continue;

            // Build a soft elliptical mask around the original box, inside the expanded ROI
            cv::Mat trainMask(expanded.size(), CV_8UC1, cv::Scalar(0));
            const cv::Rect localPieceRoi(roi.x - expanded.x, roi.y - expanded.y, roi.width, roi.height);

            const cv::Point center(localPieceRoi.x + localPieceRoi.width / 2,
                                   localPieceRoi.y + localPieceRoi.height / 2);
            const cv::Size axes(std::max(1, static_cast<int>(std::lround(localPieceRoi.width * 0.55f))),
                                std::max(1, static_cast<int>(std::lround(localPieceRoi.height * 0.55f))));
            cv::ellipse(trainMask, center, axes, 0.0, 0.0, 360.0, cv::Scalar(255), cv::FILLED);

            if (trainMask.cols > 4 && trainMask.rows > 4) {
                cv::GaussianBlur(trainMask, trainMask, cv::Size(3, 3), 0.0);
                cv::threshold(trainMask, trainMask, 1, 255, cv::THRESH_BINARY);
            }

            addTrainingSampleFromGlobalSift(pre, globalKeypoints, globalDescriptors, expanded, cat, trainMask);
        }
    }

    // -----------------------------
    // DATASET 2: clean crops per class
    // -----------------------------
    std::cout << std::endl;
    std::cout << "Training on dataset 2" << std::endl;

    const std::vector<ChessPieceCategory> categories = {BISHOP, KING, KNIGHT, PAWN, QUEEN, ROOK};
    int counter = 1;
    for (ChessPieceCategory cat: categories) {
        std::cout << "\r>>> Processing category " << counter++ << " / 6" << std::endl;
        const int n = data2.getCategorySize(cat);
        for (int i = 0; i < n; ++i) {
            cv::Mat pieceImage;
            if (!data2.getImage(cat, i, pieceImage) || pieceImage.empty()) continue;

            const cv::Mat pre = preprocessTrainingImage(pieceImage);
            if (pre.empty()) continue;

            std::vector<cv::KeyPoint> globalKeypoints;
            cv::Mat globalDescriptors;
            sift->detectAndCompute(pre, cv::noArray(), globalKeypoints, globalDescriptors);
            if (globalDescriptors.empty() || globalKeypoints.empty()) continue;

            // Whole patch mask: accept all pixels (no background class in the model)
            const cv::Mat mask(pre.size(), CV_8UC1, cv::Scalar(255));

            addTrainingSampleFromGlobalSift(
                pre,
                globalKeypoints,
                globalDescriptors,
                cv::Rect(0, 0, pre.cols, pre.rows),
                cat,
                mask
            );
        }
    }

    if (allDescriptors.empty() || metadata.empty()) {
        std::cerr << "Error: No training samples/descriptors collected." << std::endl;
        return;
    }

    // Vocabulary size
    const int k = 300;
    if (allDescriptors.rows < k) {
        std::cerr << "Error: Not enough descriptors for vocabulary size k=" << k
                << ". Collected: " << allDescriptors.rows << std::endl;
        return;
    }

    clusterCodebook(k);
    buildTrainDataset();
    trainModel();
}

// =============================================================================
// Inference (one image)
// =============================================================================
std::vector<DetectionResult> ChessClassifier::testImage(const Detection &detection) {
    std::vector<DetectionResult> outputs;

    if (detection.img.empty() || detection.extendedCellMask.empty() || model.empty() || vocabulary.empty()) {
        return outputs;
    }

    const cv::Rect imgBounds(0, 0, detection.img.cols, detection.img.rows);

    auto classifyOne = [&](const cv::Mat &extMaskFull, const cv::Mat *cellMaskFull) -> DetectionResult {
        DetectionResult det{};
        det.type = NONE;
        det.refinedBox = cv::Rect();
        det.confidence = 0.0f;

        if (extMaskFull.empty()) return det;

        // ROI from the extended mask (used to extract the patch)
        std::vector<cv::Point> extPts;
        cv::findNonZero(extMaskFull, extPts);
        if (extPts.empty()) return det;

        const cv::Rect extRoi = cv::boundingRect(extPts) & imgBounds;
        if (extRoi.width <= 1 || extRoi.height <= 1) return det;

        // Tighter ROI (optional) from the cell mask, used as a "refined" box
        cv::Rect normalRoi = extRoi;
        if (cellMaskFull && !cellMaskFull->empty()) {
            std::vector<cv::Point> cellPts;
            cv::findNonZero(*cellMaskFull, cellPts);
            if (!cellPts.empty()) {
                const cv::Rect cellRoi = cv::boundingRect(cellPts) & imgBounds;
                if (cellRoi.width > 1 && cellRoi.height > 1) normalRoi = cellRoi;
            }
        }
        det.refinedBox = normalRoi;

        // Binary ROI mask (remove grid/border with a light erosion)
        cv::Mat roiMask = extMaskFull(extRoi).clone();
        cv::threshold(roiMask, roiMask, 0, 255, cv::THRESH_BINARY);

        const int rad = std::max(1, std::min(roiMask.cols, roiMask.rows) / 24);
        const cv::Mat ker = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2 * rad + 1, 2 * rad + 1));
        cv::erode(roiMask, roiMask, ker, cv::Point(-1, -1), 1);

        // If erosion wipes too much, fall back to the original mask
        if (cv::countNonZero(roiMask) < 20) {
            roiMask = extMaskFull(extRoi).clone();
            cv::threshold(roiMask, roiMask, 0, 255, cv::THRESH_BINARY);
        }

        const int maskArea = cv::countNonZero(roiMask);
        if (maskArea <= 0) return det;

        // Preprocess patch (same pipeline used for training)
        const cv::Mat patchPre = preprocessTrainingImage(detection.img(extRoi));
        if (patchPre.empty()) return det;

        // Quick non-empty gate based on edge density inside the mask
        cv::Mat edges, maskedEdges;
        cv::Canny(patchPre, edges, 45.0, 120.0);
        cv::bitwise_and(edges, roiMask, maskedEdges);

        const float edgeDensity = static_cast<float>(cv::countNonZero(maskedEdges)) / static_cast<float>(maskArea);
        if (edgeDensity < config.kMinEdgeDensity) return det;

        // SIFT within the mask
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
        sift->detectAndCompute(patchPre, roiMask, keypoints, descriptors);
        if (descriptors.rows < config.kMinDescriptorsForPiece) return det;

        // Keep top-N descriptors by keypoint response (speed + stability)
        if (descriptors.rows > config.kPatchDescriptorBudget) {
            std::vector<int> order(descriptors.rows);
            std::iota(order.begin(), order.end(), 0);
            std::sort(order.begin(), order.end(),
                      [&](int a, int b) { return keypoints[a].response > keypoints[b].response; });

            cv::Mat reduced(config.kPatchDescriptorBudget, descriptors.cols, descriptors.type());
            std::vector<cv::KeyPoint> reducedKp;
            reducedKp.reserve(config.kPatchDescriptorBudget);

            for (int i = 0; i < config.kPatchDescriptorBudget; ++i) {
                descriptors.row(order[i]).copyTo(reduced.row(i));
                reducedKp.push_back(keypoints[order[i]]);
            }
            descriptors = reduced;
            keypoints = std::move(reducedKp);
        }

        const cv::Mat hist = computeHistogram(descriptors);
        if (hist.empty()) return det;

        // Convert votes to per-class probabilities and pick the best class
        cv::Mat votes;
        std::vector<float> probs(config.kNumModelClasses, 0.0f);
        try {
            model->getVotes(hist, votes, 0);
            probs = extractClassVoteFractions(votes, config.kNumModelClasses);
        } catch (const cv::Exception &) {
            return det;
        }

        int bestClass = 0;
        float pBest = 0.0f, pSecond = 0.0f;
        for (int c = 0; c < config.kNumModelClasses; ++c) {
            const float p = probs[c];
            if (p > pBest) {
                pSecond = pBest;
                pBest = p;
                bestClass = c;
            } else if (p > pSecond) { pSecond = p; }
        }

        det.type = modelLabelToCategory(bestClass);

        // Confidence = best prob, slightly penalized when ambiguous
        const float margin = pBest - pSecond;
        det.confidence = (margin < 0.04f) ? (pBest * 0.85f) : pBest;

        // Refine the box using the keypoint spread (in patch coordinates)
        if (!keypoints.empty()) {
            float minX = std::numeric_limits<float>::max();
            float minY = std::numeric_limits<float>::max();
            float maxX = 0.0f, maxY = 0.0f;

            for (const auto &kp: keypoints) {
                minX = std::min(minX, kp.pt.x);
                minY = std::min(minY, kp.pt.y);
                maxX = std::max(maxX, kp.pt.x);
                maxY = std::max(maxY, kp.pt.y);
            }

            const int x = extRoi.x + static_cast<int>(minX);
            const int y = extRoi.y + static_cast<int>(minY);
            const int w = std::max(1, static_cast<int>(maxX - minX + 1.0f));
            const int h = std::max(1, static_cast<int>(maxY - minY + 1.0f));

            cv::Rect kpBox(x, y, w, h);
            kpBox = (kpBox & normalRoi) & imgBounds;
            if (kpBox.width > 0 && kpBox.height > 0) det.refinedBox = kpBox;
        }

        return det;
    };

    const size_t n = detection.extendedCellMask.size();
    outputs.reserve(n);

    for (size_t i = 0; i < n; ++i) {
        const cv::Mat &extMask = detection.extendedCellMask[i];
        const cv::Mat *cellMask = (i < detection.cellMask.size()) ? &detection.cellMask[i] : nullptr;
        outputs.push_back(classifyOne(extMask, cellMask));
    }

    return outputs;
}

// =============================================================================
// Training sample builder (slice global SIFT within ROI, optionally masked)
// =============================================================================
cv::Mat ChessClassifier::addTrainingSampleFromGlobalSift(const cv::Mat &preprocessedImage,
                                                         const std::vector<cv::KeyPoint> &globalKeypoints,
                                                         const cv::Mat &globalDescriptors,
                                                         const cv::Rect &roi,
                                                         ChessPieceCategory label,
                                                         const cv::Mat &localMask) {
    if (label == NONE) return cv::Mat();
    if (preprocessedImage.empty() || globalDescriptors.empty() || globalKeypoints.empty()) return cv::Mat();

    const cv::Rect imgBounds(0, 0, preprocessedImage.cols, preprocessedImage.rows);
    const cv::Rect clipped = roi & imgBounds;
    if (clipped.width <= 1 || clipped.height <= 1) return cv::Mat();

    // Prepare a binary local mask in ROI coordinates (optional)
    cv::Mat mask;
    if (!localMask.empty()) {
        if (localMask.size() != clipped.size()) {
            cv::resize(localMask, mask, clipped.size(), 0, 0, cv::INTER_NEAREST);
        } else {
            mask = localMask.clone();
        }

        if (mask.depth() != CV_8U) {
            double mn = 0.0, mx = 0.0;
            cv::minMaxLoc(mask, &mn, &mx);
            mask.convertTo(mask, CV_8U, (mx <= 1.5) ? 255.0 : 1.0);
        } else if (mask.type() != CV_8UC1) {
            cv::cvtColor(mask, mask, cv::COLOR_BGR2GRAY);
        }

        cv::threshold(mask, mask, 0, 255, cv::THRESH_BINARY);
    }

    // Collect indices of global keypoints inside ROI and inside mask (if provided)
    std::vector<int> idx;
    idx.reserve(64);

    for (int i = 0; i < static_cast<int>(globalKeypoints.size()); ++i) {
        const auto &kp = globalKeypoints[i];
        if (!clipped.contains(kp.pt)) continue;

        if (!mask.empty()) {
            const int lx = static_cast<int>(std::lround(kp.pt.x - static_cast<float>(clipped.x)));
            const int ly = static_cast<int>(std::lround(kp.pt.y - static_cast<float>(clipped.y)));
            if (lx < 0 || ly < 0 || lx >= mask.cols || ly >= mask.rows) continue;
            if (mask.at<uchar>(ly, lx) == 0) continue;
        }

        idx.push_back(i);
    }

    if (idx.empty()) return cv::Mat();

    // Build local (ROI) descriptors + keypoints (keypoints moved to ROI coordinates)
    std::vector<cv::KeyPoint> keypoints;
    keypoints.reserve(idx.size());

    cv::Mat descriptors(static_cast<int>(idx.size()), globalDescriptors.cols, globalDescriptors.type());

    for (int r = 0; r < static_cast<int>(idx.size()); ++r) {
        const int gi = idx[r];
        cv::KeyPoint kp = globalKeypoints[gi];
        kp.pt -= cv::Point2f(static_cast<float>(clipped.x), static_cast<float>(clipped.y));
        keypoints.push_back(kp);
        globalDescriptors.row(gi).copyTo(descriptors.row(r));
    }

    // Descriptor budget: keep top responses only
    if (descriptors.rows > config.kPatchDescriptorBudget) {
        std::vector<int> order(descriptors.rows);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(),
                  [&](int a, int b) { return keypoints[a].response > keypoints[b].response; });

        cv::Mat reduced(config.kPatchDescriptorBudget, descriptors.cols, descriptors.type());
        std::vector<cv::KeyPoint> reducedKp;
        reducedKp.reserve(config.kPatchDescriptorBudget);

        for (int i = 0; i < config.kPatchDescriptorBudget; ++i) {
            descriptors.row(order[i]).copyTo(reduced.row(i));
            reducedKp.push_back(keypoints[order[i]]);
        }

        descriptors = reduced;
        keypoints = std::move(reducedKp);
    }

    // Store sample metadata for later histogram building
    TrainImageMetadata meta;
    meta.startRow = allDescriptors.rows;
    meta.numRows = descriptors.rows;
    meta.type = label;
    metadata.push_back(meta);

    allDescriptors.push_back(descriptors);
    return descriptors;
}

// =============================================================================
// Global k-means (visual vocabulary)
// =============================================================================
void ChessClassifier::clusterCodebook(int k) {
    if (allDescriptors.empty()) {
        std::cerr << "Error: No descriptors available for clustering." << std::endl;
        return;
    }

    std::cout << "***********************************************\n";
    std::cout << "**** Starting Global K-Means (" << k << " clusters)...\n";
    std::cout << "**** Total descriptors: " << allDescriptors.rows << "\n";

    cv::TermCriteria criteria(cv::TermCriteria::MAX_ITER | cv::TermCriteria::EPS,
                              config.kGlobalKMeansMaxIter, config.kGlobalKMeansEps);

    int sampleRows = static_cast<int>(std::lround(allDescriptors.rows * config.kGlobalKMeansSubsampleRatio));
    sampleRows = std::max(k, sampleRows);
    sampleRows = std::min(sampleRows, config.kMaxDescriptorsForGlobalKMeans);
    sampleRows = std::min(sampleRows, allDescriptors.rows);

    cv::Mat sample;
    if (sampleRows >= allDescriptors.rows) {
        sample = allDescriptors;
    } else {
        sample = cv::Mat(sampleRows, allDescriptors.cols, allDescriptors.type());
        cv::RNG rng(static_cast<uint64>(cv::getTickCount()));
        for (int i = 0; i < sampleRows; ++i) {
            const int srcRow = rng.uniform(0, allDescriptors.rows);
            allDescriptors.row(srcRow).copyTo(sample.row(i));
        }
    }

    std::cout << "**** Subsampled for k-means: " << sample.rows << " / " << allDescriptors.rows << "\n";

    std::vector<int> labels; // required by cv::kmeans
    const clock_t t0 = clock();

    cv::kmeans(sample, k, labels, criteria, config.kGlobalKMeansAttempts,
               cv::KMEANS_PP_CENTERS, vocabulary);

    const double secs = (clock() - t0) / double(CLOCKS_PER_SEC);
    std::cout << "**** K-Means finished in " << secs << " s\n";
    std::cout << "**** Vocabulary: [" << vocabulary.rows << " x " << vocabulary.cols << "]\n";
    std::cout << "***********************************************\n";
}

// =============================================================================
// Build (BoW) train matrix + labels
// =============================================================================
void ChessClassifier::buildTrainDataset() {
    if (vocabulary.empty() || allDescriptors.empty() || metadata.empty()) {
        std::cerr << "Error: Missing data to build training dataset." << std::endl;
        trainData.release();
        trainLabels.release();
        return;
    }

    const int numSamples = static_cast<int>(metadata.size());
    const int bins = vocabulary.rows;

    trainData = cv::Mat::zeros(numSamples, bins, CV_32F);
    trainLabels = cv::Mat::ones(numSamples, 1, CV_32S) * -1;

    for (int i = 0; i < numSamples; ++i) {
        const TrainImageMetadata &meta = metadata[i];
        const int endRow = meta.startRow + meta.numRows;
        if (meta.numRows <= 0 || meta.startRow < 0 || endRow > allDescriptors.rows) continue;

        const int lbl = categoryToModelLabel(meta.type);
        if (lbl < 0 || lbl >= config.kNumModelClasses) continue;

        const cv::Mat desc = allDescriptors.rowRange(meta.startRow, endRow);
        const cv::Mat hist = computeHistogram(desc);
        if (hist.empty()) continue;

        hist.copyTo(trainData.row(i));
        trainLabels.at<int>(i, 0) = lbl;
    }

    // Print class distribution
    std::vector<int> classCounts(config.kNumModelClasses, 0);
    for (int i = 0; i < trainLabels.rows; ++i) {
        const int lbl = trainLabels.at<int>(i, 0);
        if (0 <= lbl && lbl < config.kNumModelClasses) ++classCounts[lbl];
    }

    std::cout << "Training class distribution [0..5]: ";
    for (int c = 0; c < config.kNumModelClasses; ++c) {
        std::cout << classCounts[c] << (c + 1 < config.kNumModelClasses ? " " : "");
    }
    std::cout << "\n";
}

// =============================================================================
// Train Random Forest (balanced per class)
// =============================================================================
void ChessClassifier::trainModel() {
    if (trainData.empty() || trainLabels.empty()) {
        std::cerr << "Error: Missing training data." << std::endl;
        model.release();
        return;
    }

    // Filter invalid labels (-1)
    cv::Mat filteredData, filteredLabels;
    for (int i = 0; i < trainLabels.rows; ++i) {
        const int lbl = trainLabels.at<int>(i, 0);
        if (0 <= lbl && lbl < config.kNumModelClasses) {
            filteredData.push_back(trainData.row(i));
            filteredLabels.push_back(trainLabels.row(i));
        }
    }
    if (filteredData.empty()) {
        std::cerr << "Error: No valid labeled samples for training." << std::endl;
        model.release();
        return;
    }

    // Group indices by class
    std::vector<std::vector<int> > byClass(config.kNumModelClasses);
    for (int i = 0; i < filteredLabels.rows; ++i) {
        const int lbl = filteredLabels.at<int>(i, 0);
        byClass[lbl].push_back(i);
    }

    // Balance to the smallest non-empty class
    int minCount = std::numeric_limits<int>::max();
    for (int c = 0; c < config.kNumModelClasses; ++c) {
        if (!byClass[c].empty()) minCount = std::min(minCount, static_cast<int>(byClass[c].size()));
    }
    if (minCount == std::numeric_limits<int>::max()) {
        std::cerr << "Error: Missing classes after filtering." << std::endl;
        model.release();
        return;
    }

    const int targetPerClass = minCount;

    cv::RNG rng(static_cast<uint64>(cv::getTickCount()));
    cv::Mat balancedData, balancedLabels;

    for (int c = 0; c < config.kNumModelClasses; ++c) {
        auto &idx = byClass[c];
        if (idx.empty()) continue;

        cv::randShuffle(idx, 1.0, &rng);
        const int take = std::min(targetPerClass, static_cast<int>(idx.size()));
        for (int k = 0; k < take; ++k) {
            balancedData.push_back(filteredData.row(idx[k]));
            balancedLabels.push_back(filteredLabels.row(idx[k]));
        }
    }

    std::cout << "Balanced training set size: " << balancedLabels.rows
            << " (target/class=" << targetPerClass << ")\n";

    if (balancedData.empty()) {
        std::cerr << "Error: Failed to build balanced training set." << std::endl;
        model.release();
        return;
    }

    model = cv::ml::RTrees::create();
    model->setMaxDepth(16);
    model->setMinSampleCount(2);
    model->setRegressionAccuracy(0.0f);
    model->setUseSurrogates(false);
    model->setMaxCategories(config.kNumModelClasses);
    model->setCalculateVarImportance(false);
    model->setActiveVarCount(0);
    model->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER, 200, 0));

    if (!model->train(balancedData, cv::ml::ROW_SAMPLE, balancedLabels)) {
        std::cerr << "Error: Random Forest training failed." << std::endl;
        model.release();
    }
}

// =============================================================================
// BoW histogram (soft outlier rejection by distance median)
// =============================================================================
cv::Mat ChessClassifier::computeHistogram(const cv::Mat &descriptors) const {
    if (vocabulary.empty()) return cv::Mat();

    cv::Mat hist = cv::Mat::zeros(1, vocabulary.rows, CV_32F);
    if (descriptors.empty()) return hist;

    cv::BFMatcher matcher(cv::NORM_L2);
    std::vector<cv::DMatch> matches;
    matcher.match(descriptors, vocabulary, matches);
    if (matches.empty()) return hist;

    // Robust distance threshold using median
    std::vector<float> dists;
    dists.reserve(matches.size());
    for (const auto &m: matches) dists.push_back(m.distance);

    std::nth_element(dists.begin(), dists.begin() + dists.size() / 2, dists.end());
    const float med = dists[dists.size() / 2];
    const float thr = std::max(1e-6f, 1.5f * med);

    for (const auto &m: matches) {
        if (m.distance <= thr) hist.at<float>(0, m.trainIdx) += 1.0f;
    }

    // L1 normalize: histogram sums to 1
    cv::normalize(hist, hist, 1.0, 0.0, cv::NORM_L1);
    return hist;
}

// =============================================================================
// Confidence helper
// =============================================================================
float ChessClassifier::computeClassificationConfidence(const cv::Mat &histogram) const {
    if (model.empty() || histogram.empty()) return 0.0f;

    try {
        cv::Mat votes;
        model->getVotes(histogram, votes, 0);
        const auto probs = extractClassVoteFractions(votes, config.kNumModelClasses);

        float best = 0.0f;
        for (float p: probs) best = std::max(best, p);
        return std::max(0.0f, std::min(best, 1.0f));
    } catch (const cv::Exception &) {
        return 0.0f;
    }
}

cv::Mat ChessClassifier::preprocessTrainingImage(const cv::Mat &img, int kernelSize, double sigma) {
    if (img.empty()) return {};

    if (kernelSize <= 0) kernelSize = 1;
    if (kernelSize % 2 == 0) ++kernelSize;

    cv::Mat gray;
    if (img.channels() == 1) gray = img.clone();
    else cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

    if (kernelSize > 1) {
        double s = (sigma > 0.0) ? sigma : 0.0; // 0 => OpenCV chooses
        cv::GaussianBlur(gray, gray, cv::Size(kernelSize, kernelSize), s);
    }

    cv::Scalar mean, stddev;
    cv::meanStdDev(gray, mean, stddev);

    if (stddev[0] < 25.0) {
        auto clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
        cv::Mat out;
        clahe->apply(gray, out);
        return out;
    }

    return gray;
}

float ChessClassifier::voteValueAt(const cv::Mat &votes, const int r, const int c) {
    switch (votes.type()) {
        case CV_32S: return static_cast<float>(votes.at<int>(r, c));
        case CV_32F: return votes.at<float>(r, c);
        case CV_64F: return static_cast<float>(votes.at<double>(r, c));
        default: return 0.0f;
    }
}

std::vector<float> ChessClassifier::extractClassVoteFractions(const cv::Mat &votes, const int numClasses) {
    std::vector<float> fractions(numClasses, 0.0f);
    if (votes.empty() || numClasses <= 0) return fractions;

    std::vector<float> classVotes(numClasses, 0.0f);

    // Common RTrees format: 2 x K (row0=labels, row1=votes)
    if (votes.rows == 2 && votes.cols >= 1) {
        for (int c = 0; c < votes.cols; ++c) {
            const int label = static_cast<int>(std::lround(voteValueAt(votes, 0, c)));
            const float v = std::max(0.0f, voteValueAt(votes, 1, c));
            if (0 <= label && label < numClasses) classVotes[label] += v;
        }
    }
    // Alternate format: N x 2 (each row = [label, vote])
    else if (votes.cols == 2 && votes.rows >= 1) {
        for (int r = 0; r < votes.rows; ++r) {
            const int label = static_cast<int>(std::lround(voteValueAt(votes, r, 0)));
            const float v = std::max(0.0f, voteValueAt(votes, r, 1));
            if (0 <= label && label < numClasses) classVotes[label] += v;
        }
    }
    // Fallback: treat as a flat vector and read the first numClasses entries
    else {
        const int n = std::min(numClasses, static_cast<int>(votes.total()));
        for (int i = 0; i < n; ++i) {
            float v = 0.0f;
            if (votes.type() == CV_32S) v = static_cast<float>(votes.at<int>(i));
            else if (votes.type() == CV_32F) v = votes.at<float>(i);
            else if (votes.type() == CV_64F) v = static_cast<float>(votes.at<double>(i));
            classVotes[i] = std::max(0.0f, v);
        }
    }

    float sumVotes = 0.0f;
    for (float v: classVotes) sumVotes += std::max(0.0f, v);
    if (sumVotes <= 0.0f) return fractions;

    for (int i = 0; i < numClasses; ++i) fractions[i] = std::max(0.0f, classVotes[i]) / sumVotes;
    return fractions;
}

// -----------------------------------------------------------------------------
// Small utilities for ROIs
// -----------------------------------------------------------------------------
cv::Rect ChessClassifier::clampRoi(const cv::Rect &roi, const cv::Size &size) {
    return roi & cv::Rect(0, 0, size.width, size.height);
}

cv::Rect ChessClassifier::expandRoi(const cv::Rect &roi, const cv::Size &size, float scale) {
    const int cx = roi.x + roi.width / 2;
    const int cy = roi.y + roi.height / 2;
    const int newW = std::max(2, static_cast<int>(std::lround(static_cast<float>(roi.width) * scale)));
    const int newH = std::max(2, static_cast<int>(std::lround(static_cast<float>(roi.height) * scale)));
    return clampRoi(cv::Rect(cx - newW / 2, cy - newH / 2, newW, newH), size);
}
