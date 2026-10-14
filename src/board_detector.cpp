// Author: Leonardo Gusson, 2157199

#include <random>
#include "geometry_utils.h"
#include "board_detector.h"

#include "io_functions.h"

int BoardDetector::DEBUG_window_index = 0;

Detection BoardDetector::detect(const cv::Mat &img) const {
    // BGR to GRAY conversion
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

    // *********************************************************************************
    // Find all line segments with LSD detector
    const cv::Ptr<cv::LineSegmentDetector> lsd = cv::createLineSegmentDetector(cv::LSD_REFINE_STD);
    std::vector<cv::Vec4f> allSegments;
    lsd->detect(gray, allSegments);

    // Remove lines with length < x% of the image wider dim
    std::vector<cv::Vec4f> filteredSegments;
    const double minLen = std::max(gray.cols, gray.rows) * config.minSegmentLengthRatio;
    for (const auto &line: allSegments) {
        const double length = cv::norm(
            cv::Point2f(line[0], line[1]) - cv::Point2f(line[2], line[3])
        );
        if (length > minLen) {
            filteredSegments.push_back(line);
        }
    }

    // *********************************************************************************
    // Merge aligned segments (strict thresholds to avoid noise)
    std::vector<CollinearSegments> lineClusters = mergeAlignedSegments(filteredSegments);
    removeCollinearClusters(lineClusters);
    if (config.debug1) DEBUG_print1(gray, lsd, allSegments, filteredSegments);

    // *********************************************************************************
    // Apply conditions to filter out noise segments
    geometricFilter(lineClusters, std::max(gray.rows, gray.cols));
    if (config.debug2) DEBUG_print2(gray, lineClusters);

    // *********************************************************************************
    // Angular clustering
    std::vector<CollinearSegments> horClusters, vertClusters;
    angularClustering(lineClusters, horClusters, vertClusters);
    angularFilter(horClusters);
    angularFilter(vertClusters);
    if (config.debug3a) DEBUG_print3(gray, horClusters, vertClusters, 2.5, "angularFiltered");

    // *********************************************************************************
    // Spatial filtering
    std::vector<double> horClusterSpacing, vertClusterSpacing;
    sortLinesSpatially(horClusters, vertClusters, gray.size(), horClusterSpacing, vertClusterSpacing);
    double avgHorSpace, avgVertSpace;
    spatialGridFilter(horClusters, vertClusters,
                      horClusterSpacing, vertClusterSpacing,
                      gray.size(), avgHorSpace, avgVertSpace);
    if (config.debug3b) DEBUG_print3(gray, horClusters, vertClusters, 1, "spatialFiltered");

    // *********************************************************************************
    // Grid extraction
    std::vector<std::vector<cv::Point2f> > squaresCorners = {};
    extractGrid(horClusters, vertClusters, squaresCorners);

    // *********************************************************************************
    // Check occupation
    std::vector<bool> emptySquares(squaresCorners.size());
    Detection detection;
    getCandidatePieces(img, gray, squaresCorners, avgHorSpace, avgVertSpace, emptySquares, detection);
    detection.isFirstWhite = estimateTopLeftColor(gray, squaresCorners, emptySquares);
    if (config.debug4) DEBUG_print4(gray, squaresCorners, emptySquares);

    DEBUG_window_index++;
    return detection;
}

/***********************************************************************************/
/******************************* GEOMETRIC FILTERING *******************************/
/***********************************************************************************/

std::vector<CollinearSegments> BoardDetector::mergeAlignedSegments(const std::vector<cv::Vec4f> &segments) const {
    std::vector<CollinearSegments> clusters;
    for (const cv::Vec4f &segment: segments) {
        // Compute the line associated to the current segment
        cv::Vec3f line = GeometryUtils::pointsToLineABC(segment);
        const float norm = std::hypot(line[0], line[1]);
        cv::Vec2f dir(-line[1] / norm, line[0] / norm);

        // Group initialization
        if (clusters.empty()) {
            clusters.emplace_back(line, dir, segment);
            continue;
        }

        // Compare this line with prev. seen groups
        bool assignedToPrevCluster = false;
        for (auto &cluster: clusters) {
            // angular threshold for parallelism check: 5deg; point-line distance threshold: 5px
            if (GeometryUtils::areCollinear(
                    cluster.line,
                    segment,
                    config.parallelismAngleThreshold,
                    config.collinearityDistThreshold)
            ) {
                cluster.addSegment(segment);
                assignedToPrevCluster = true;
            }
        }

        // If the segment does not belong to any prev. seen line, create a new group
        if (!assignedToPrevCluster) {
            clusters.emplace_back(line, dir, segment);
        }
    }

    // After all clusters are populated, initialize segments' projections on each cluster's reference line
    for (auto &cluster: clusters) cluster.project();

    return clusters;
}

void BoardDetector::removeCollinearClusters(std::vector<CollinearSegments> &clusters) const {
    if (clusters.size() < 2) return;

    std::vector<CollinearSegments> newClusters = {clusters[0]};
    for (size_t i = 1; i < clusters.size(); ++i) {
        bool isUnique = true;

        const double sampleX = (clusters[i].segments.front()[0] + clusters[i].segments.front()[2]) / 2;
        const double sampleY = (clusters[i].segments.front()[1] + clusters[i].segments.front()[3]) / 2;
        const auto sample = cv::Point2f(sampleX, sampleY);

        for (auto &newCluster: newClusters) {
            if (GeometryUtils::areParallel(clusters[i].line, newCluster.line, config.mergeCollinearAngleThreshold)
                && GeometryUtils::pointLineDist(sample, newCluster.line) < config.mergeCollinearDistThreshold) {
                // merge clusters[i] into newCluster
                CollinearSegments::merge(newCluster, clusters[i]);
                isUnique = false;
                break;
            }
        }
        if (isUnique) newClusters.push_back(clusters[i]);
    }

    clusters = newClusters;
}

void BoardDetector::geometricFilter(std::vector<CollinearSegments> &clusters, const int imgMaxDim) const {
    std::vector<CollinearSegments> filteredClusters;

    // Filters
    for (CollinearSegments &cluster: clusters) {
        // Few segments and scarse coverage => discard
        const bool fewOccurrences = cluster.projection.intervals.size() < config.minSegmentsPerCluster;
        const bool highCoverage = cluster.projection.sumIntervals() > config.highLineCoverageRatio * imgMaxDim;
        if (fewOccurrences && !highCoverage) continue;
        // else, few segments and high coverage => ok
        // else, enough occurrences but scarse coverage => ok (dangerous case, further filtered below)
        // else, enough occurrences and high coverage => ok (best case)

        // Poor coverage => discard line
        const bool poorCoverage = cluster.projection.sumIntervals() < config.minLineCoverageRatio * imgMaxDim;
        if (poorCoverage) continue;

        // Extremely high coverage => discard line (noise)
        if (cluster.projection.sumIntervals() > 0.95 * imgMaxDim) continue;

        // High max gap => discard
        if (cluster.projection.maxGap() > config.maxGapRatio * imgMaxDim) continue;
        if (cluster.projection.avgGap() > config.maxAvgGapRatio * imgMaxDim) continue;

        filteredClusters.push_back(cluster);
    }

    clusters = filteredClusters;
}

/**********************************************************************************/
/******************************* ANGULAR FILTERING ********************************/
/**********************************************************************************/

void BoardDetector::angularClustering(std::vector<CollinearSegments> &clusters,
                                      std::vector<CollinearSegments> &horClusters,
                                      std::vector<CollinearSegments> &vertClusters) const {
    if (clusters.size() < 2) {
        std::cerr << "Can't perform kmeans (K=2) on " << clusters.size() << " lines." << std::endl;
        return;
    }

    // Compute the absolute angles of each cluster's reference line
    computeClustersAngle(clusters);

    // Set up the dataset for the angular k-means clustering
    // Every line's angle is doubled to cope with 0-pi discontinuity
    // Then it is mapped to a circle to perform k-means clusetering in 2-dims
    cv::Mat dataset(clusters.size(), 2, CV_32F);
    for (size_t i = 0; i < clusters.size(); ++i) {
        dataset.at<float>(i, 0) = std::cos(clusters[i].angle * 2.0);
        dataset.at<float>(i, 1) = std::sin(clusters[i].angle * 2.0);
    }

    // K-means
    cv::Mat labels, centers;
    const cv::TermCriteria criteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 10, 0.1);
    cv::kmeans(dataset, 2, labels, criteria, 5, cv::KMEANS_PP_CENTERS, centers);

    // Read centroids
    float angles[2];
    float distFromVertical[2];
    for (int k = 0; k < 2; ++k) {
        float angle = 0.5f * std::atan2(centers.at<float>(k, 1), centers.at<float>(k, 0));
        if (angle < 0) angle += CV_PI;
        angles[k] = angle;

        float diff = std::abs(angle - CV_PI / 2.0f);
        if (diff > CV_PI / 2) diff = CV_PI - diff;
        distFromVertical[k] = diff;
    }
    const int verticalLabel = (distFromVertical[0] < distFromVertical[1]) ? 0 : 1;

    // Filter cluster lines discarding those differing more than config.angularThreshold from its centroid
    horClusters.clear();
    vertClusters.clear();
    for (size_t i = 0; i < clusters.size(); ++i) {
        const int label = labels.at<int>(i);
        const float centroid = angles[label];

        double diff = std::abs(clusters[i].angle - centroid);
        if (diff > CV_PI / 2) diff = CV_PI - diff;

        if (diff > config.maxClusterAngleDeviation)
            continue;

        if (label == verticalLabel) {
            vertClusters.push_back(clusters[i]);
        } else {
            horClusters.push_back(clusters[i]);
        }
    }
}

void BoardDetector::computeClustersAngle(std::vector<CollinearSegments> &clusters) {
    for (auto &cluster: clusters) {
        double ang = std::atan2(cluster.dir[1], cluster.dir[0]); // [-pi, pi]
        if (ang < 0) ang += CV_PI;
        cluster.angle = ang;
    }
}

void BoardDetector::angularFilter(std::vector<CollinearSegments> &lines) const {
    if (lines.empty()) return;

    // Compute angles mean and stddev
    std::vector<double> angles;
    angles.reserve(lines.size());
    for (const auto &c: lines)
        angles.push_back((c.angle));


    double sumSin = 0.0;
    double sumCos = 0.0;
    for (const double theta: angles) {
        sumSin += std::sin(2 * theta);
        sumCos += std::cos(2 * theta);
    }

    // mean
    double mean = std::atan2(sumSin, sumCos) / 2.0;
    if (mean < 0) mean += CV_PI;

    // stddev
    double R = std::sqrt(sumSin * sumSin + sumCos * sumCos) / angles.size();
    if (R > 1.0) R = 1.0;
    double stddev = 0.0;
    if (R > std::numeric_limits<double>::epsilon()) {
        stddev = std::sqrt(-2.0 * std::log(R)) / 2.0;
    } else {
        stddev = std::numeric_limits<double>::infinity();
    }

    // Case 1: almost parallel
    std::vector<CollinearSegments> filtered;
    if (stddev < config.maxParallelStdDev) {
        for (size_t i = 0; i < angles.size(); ++i) {
            double diff = std::abs(mean - angles[i]);
            diff = std::min(diff, CV_PI - diff);
            if (diff > std::max(config.sigmaFilter * stddev, 0.017)) {
                continue;
            }
            filtered.push_back(lines[i]);
        }
    }
    // Case 2: converging lines
    else {
        // Search the best vanishing point
        std::vector<bool> inliers(lines.size(), false);
        int maxInliers = 0;
        cv::Point2f bestVP;

        // select every pair of lines and compare their intersection (as vp candidate) with other lines
        // this assumes lines.size() is small. in our case it is typically < 10-12
        const int nLines = static_cast<int>(lines.size());
        for (int i = 0; i < nLines; ++i) {
            for (int j = i + 1; j < nLines; ++j) {
                cv::Point2f vp = GeometryUtils::findIntersection(lines[i].line, lines[j].line);
                if (vp.x == FLT_MAX || vp.y == FLT_MAX || std::abs(vp.x) > 50000 || std::abs(vp.y) > 50000) continue;

                // Count inliers (other lines converging to this vp candidate)
                int currentInliers = 0;
                for (const auto &l: lines) {
                    if (GeometryUtils::pointLineDist(vp, l.line) < config.vanishingPointDistThreshold)
                        currentInliers++;
                }

                // Update bestVP
                if (currentInliers > maxInliers) {
                    maxInliers = currentInliers;
                    bestVP = vp;
                }
            }
        }

        // Re-compute the bestVP's inliers to filter lines vector
        if (maxInliers > 0) {
            for (size_t k = 0; k < lines.size(); ++k) {
                if (GeometryUtils::pointLineDist(bestVP, lines[k].line) < config.vanishingPointDistThreshold) {
                    inliers[k] = true;
                }
            }
            for (size_t k = 0; k < lines.size(); ++k) {
                if (inliers[k]) filtered.push_back(lines[k]);
            }
        }
    }

    if (!filtered.empty())
        lines = filtered;
}

/**********************************************************************************/
/******************************* SPATIAL FILTERING ********************************/
/**********************************************************************************/

void BoardDetector::sortLinesSpatially(std::vector<CollinearSegments> &horClusters,
                                       std::vector<CollinearSegments> &vertClusters,
                                       const cv::Size &imgSize, std::vector<double> &horSpacing,
                                       std::vector<double> &vertSpacing) {
    // Image center
    const double midX = imgSize.width / 2.0f;
    const double midY = imgSize.height / 2.0f;

    // Temporary association between clusters and spacing vectors
    std::vector<std::pair<CollinearSegments, double> > tempHorPairs, tempVertPairs;
    tempHorPairs.reserve(horClusters.size());
    tempVertPairs.reserve(vertClusters.size());

    // Compute spacing
    // For vertical lines, spacing contains the x coordinate of each line when y = midY
    for (auto &vertCluster: vertClusters) {
        // x = (-by - c) / a
        double x = (-vertCluster.line[1] * midY - vertCluster.line[2]) / (vertCluster.line[0] + 1e-6);
        tempVertPairs.emplace_back(std::move(vertCluster), x);
    }
    // For horizontal lines, spacing contains the y coordinate of each line when x = midX
    for (auto &horCluster: horClusters) {
        // y = (-ax - c) / b
        double y = (-horCluster.line[0] * midX - horCluster.line[2]) / (horCluster.line[1] + 1e-6);
        tempHorPairs.emplace_back(std::move(horCluster), y);
    }

    // Sort vertical clusters (left to right) and horizontal clusters (top to bottom)
    std::sort(tempVertPairs.begin(), tempVertPairs.end(),
              [](const std::pair<CollinearSegments, double> &pairA,
                 const std::pair<CollinearSegments, double> &pairB) {
                  return pairA.second < pairB.second;
              });
    std::sort(tempHorPairs.begin(), tempHorPairs.end(),
              [](const std::pair<CollinearSegments, double> &pairA,
                 const std::pair<CollinearSegments, double> &pairB) {
                  return pairA.second < pairB.second;
              });

    // Restore original form
    horClusters.clear();
    vertClusters.clear();
    horSpacing.clear();
    vertSpacing.clear();
    horClusters.reserve(tempHorPairs.size());
    vertClusters.reserve(tempVertPairs.size());
    horSpacing.reserve(tempHorPairs.size());
    vertSpacing.reserve(tempVertPairs.size());
    for (auto &p: tempHorPairs) {
        horClusters.push_back(std::move(p.first));
        horSpacing.push_back(p.second);
    }
    for (auto &p: tempVertPairs) {
        vertClusters.push_back(std::move(p.first));
        vertSpacing.push_back(p.second);
    }
}

bool BoardDetector::spatialGridFilter(std::vector<CollinearSegments> &horClusters,
                                      std::vector<CollinearSegments> &vertClusters,
                                      std::vector<double> &horSpacing, std::vector<double> &vertSpacing,
                                      const cv::Size &imgSize, double &avgHorSpacing, double &avgVertSpacing) const {
    if (horSpacing.size() <= 1 || vertSpacing.size() <= 1) return false;

    // Compute gaps between consecutive lines in each cluster
    std::vector<double> horGaps = {}, vertGaps = {};
    horGaps.reserve(horSpacing.size() - 1);
    vertGaps.reserve(vertSpacing.size() - 1);
    for (size_t i = 1; i < horSpacing.size(); ++i) // gap[0]: spacing[0] <---> spacing[1]
        horGaps.push_back(std::abs(horSpacing[i] - horSpacing[i - 1]));
    for (size_t i = 1; i < vertSpacing.size(); ++i)
        vertGaps.push_back(std::abs(vertSpacing[i] - vertSpacing[i - 1]));

    std::vector<int> horSeeds, vertSeeds;
    const bool horSuccess = getSeeds(horGaps, horSeeds, avgHorSpacing);
    const bool vertSuccess = getSeeds(vertGaps, vertSeeds, avgVertSpacing);

    if (!horSuccess || !vertSuccess) return false; // Can't find a valid chessboard

    // each seed vector represents (at least) 3 consecutive, parallel, board lines
    // other lines delimiting a gap similar to the seeds, should be kept only if their distance
    // is an integer multiple of the seed-delimited distance
    std::vector<CollinearSegments> filteredHorClusters, filteredVertClusters;
    std::vector<double> filteredHorSpacing, filteredVertSpacing;

    std::vector<bool> horFilter, vertFilter;
    expandSeeds(horSpacing, horSeeds, imgSize.width, horFilter);
    expandSeeds(vertSpacing, vertSeeds, imgSize.height, vertFilter);

    // Apply filter and return
    for (size_t i = 0; i < horFilter.size(); ++i) {
        if (horFilter[i]) {
            filteredHorClusters.push_back(std::move(horClusters[i]));
            filteredHorSpacing.push_back(horSpacing[i]);
        }
    }
    for (size_t i = 0; i < vertFilter.size(); ++i) {
        if (vertFilter[i]) { filteredVertClusters.push_back(std::move(vertClusters[i])); }
        filteredVertSpacing.push_back(vertSpacing[i]);
    }
    horClusters = filteredHorClusters;
    vertClusters = filteredVertClusters;
    horSpacing = filteredHorSpacing;
    vertSpacing = filteredVertSpacing;
    return true;
}

bool BoardDetector::getSeeds(const std::vector<double> &gaps, std::vector<int> &seeds, double &avgGap
) const {
    struct GapGroup {
        std::vector<int> indexes;
        double bestGapApprox;
    };

    std::vector<GapGroup> groups = {};
    groups.reserve(gaps.size());
    for (int current = 0; current < static_cast<int>(gaps.size()); ++current) {
        groups.push_back(GapGroup{{current}, gaps[current]});
        for (int i = 0; i < static_cast<int>(gaps.size()); ++i) {
            if (i == current) continue;

            // If this gap is close to the dynamic average & not too far from the current (original) gap
            const bool closeToAvg = std::abs(groups.back().bestGapApprox - gaps[i]) < groups.back().bestGapApprox *
                                    config.gapSimilarityThreshold;
            const bool closeToCurrentGap = std::abs(gaps[current] - gaps[i]) < (
                                               gaps[current] * (config.gapSimilarityThreshold + 0.2));
            if (closeToAvg && closeToCurrentGap) {
                groups.back().indexes.push_back(i);
                groups.back().bestGapApprox += (1.0 / groups.back().indexes.size()) * (
                    gaps[i] - groups.back().bestGapApprox);
            }
        }
    }
    std::sort(groups.begin(), groups.end(),
              [](const GapGroup &a, const GapGroup &b) { return a.indexes.size() > b.indexes.size(); });

    // can't find a valid grid (chessboard)
    if (groups.empty()) return false;

    // filter out inconsistent lines
    std::sort(groups.front().indexes.begin(), groups.front().indexes.end(),
              [](const int a, const int b) { return a < b; });
    const std::vector<int> tempSeeds = getConsecutive(groups.front().indexes);


    // can't find a valid grid (chessboard)
    if (tempSeeds.size() < config.minGapsCount) {
        return false;
    } else {
        seeds.clear();
        const int mid = tempSeeds.size() / 2;
        seeds.push_back(tempSeeds[mid - 1]);
        seeds.push_back(tempSeeds[mid]);
    }
    avgGap = groups.front().bestGapApprox;
    return true;
}

std::vector<int> BoardDetector::getConsecutive(const std::vector<int> &v) {
    std::vector<int> best = {};
    std::vector<int> current = {};
    if (v.empty()) return best;

    current.push_back(v[0]);
    for (size_t i = 1; i < v.size(); ++i) {
        if (v[i] == v[i - 1] + 1) {
            current.push_back(v[i]);
        } else {
            if (current.size() > best.size()) best = current;
            current.clear();
            current.push_back(v[i]);
        }
    }
    if (current.size() > best.size()) best = current;
    return best;
}

void BoardDetector::expandSeeds(const std::vector<double> &spacing, const std::vector<int> &seedGapIndexes,
                                int imgExtension, std::vector<bool> &outFilter) const {
    if (seedGapIndexes.empty()) return;
    // Initialize the output filter with the lines enclosing seed gaps
    int minSeedLineIdx = seedGapIndexes.front();
    int maxSeedLineIdx = seedGapIndexes.back() + 1;
    outFilter.assign(spacing.size(), false);
    for (int i = minSeedLineIdx; i <= maxSeedLineIdx; ++i)
        outFilter[i] = true;

    // Enough lines
    constexpr int targetLines = 9;
    if (std::count(outFilter.begin(), outFilter.end(), true) >= targetLines) return;

    // Expansion lambdas
    auto oneStepExpand = [this, &spacing, &outFilter](bool backward, int &minSeedLineIdx, int &maxSeedLineIdx,
                                                      double &backwardCurrentGap, double &forwardCurrentGap) {
        if (backward) {
            int currentIdx = minSeedLineIdx;
            if (currentIdx == 0) return false;
            const double startingSpacing = spacing[currentIdx];

            double dist = 0; // distance moved during this backward expansion
            do {
                currentIdx--;
                dist = startingSpacing - spacing[currentIdx];

                // expansion
                if (std::abs(dist - backwardCurrentGap) < backwardCurrentGap * config.gapExpansionToleranceThreshold) {
                    if (minSeedLineIdx - currentIdx > 1)
                        return false; // Chessboard border or noise

                    backwardCurrentGap = dist;
                    outFilter[currentIdx] = true;
                    minSeedLineIdx = currentIdx;
                    return true;
                }
            } while (currentIdx > 0 && dist < 1.5 * backwardCurrentGap);
            return false;
        } else {
            int currentIdx = maxSeedLineIdx;
            if (currentIdx == spacing.size() - 1) return false;
            const double startingSpacing = spacing[currentIdx];

            double dist = 0;
            do {
                currentIdx++;
                dist = spacing[currentIdx] - startingSpacing;

                // expansion
                if (std::abs(dist - forwardCurrentGap) < forwardCurrentGap * config.gapExpansionToleranceThreshold) {
                    if (currentIdx - maxSeedLineIdx > 1) return false; // Chessboard border or noise

                    forwardCurrentGap = dist;
                    outFilter[currentIdx] = true;
                    maxSeedLineIdx = currentIdx;
                    return true;
                }
            } while (currentIdx < spacing.size() - 1 && dist < 1.5 * forwardCurrentGap);
            return false;
        }
    };

    // Decide the expansion direction
    // Note: spacing vectors contains image coordinates on the lines x=midX and y=midY
    double backwardMargin = spacing[minSeedLineIdx];
    double forwardMargin = imgExtension - spacing[maxSeedLineIdx];
    double backwardCurrentGap = spacing[minSeedLineIdx + 1] - spacing[minSeedLineIdx];
    double forwardCurrentGap = spacing[maxSeedLineIdx] - spacing[maxSeedLineIdx - 1];
    bool canExpandBackward = true;
    bool canExpandForward = true;

    while (std::count(outFilter.begin(), outFilter.end(), true) < targetLines) {
        if (!canExpandBackward && !canExpandForward) break;

        bool tryBackward = (backwardMargin > forwardMargin);
        if (tryBackward && !canExpandBackward) tryBackward = false;
        if (!tryBackward && !canExpandForward) tryBackward = true;

        if (tryBackward) {
            if (!oneStepExpand(true, minSeedLineIdx, maxSeedLineIdx, backwardCurrentGap, forwardCurrentGap)) {
                canExpandBackward = false;
            }
        } else {
            if (!oneStepExpand(false, minSeedLineIdx, maxSeedLineIdx, backwardCurrentGap, forwardCurrentGap)) {
                canExpandForward = false;
            }
        }

        backwardMargin = spacing[minSeedLineIdx];
        forwardMargin = imgExtension - spacing[maxSeedLineIdx];
    }
}

/**********************************************************************************/
/********************************* GRID EXTRACTION ********************************/
/**********************************************************************************/

void BoardDetector::extractGrid(const std::vector<CollinearSegments> &horLines,
                                const std::vector<CollinearSegments> &vertLines,

                                std::vector<std::vector<cv::Point2f> > &squaresCorners
) {
    std::vector<std::vector<cv::Point2f> > intersections(
        horLines.size(),
        std::vector<cv::Point2f>(vertLines.size()));


    for (size_t i = 0; i < horLines.size(); ++i) {
        for (size_t j = 0; j < vertLines.size(); ++j) {
            intersections[i][j] = GeometryUtils::findIntersection(horLines[i].line, vertLines[j].line);
        }
    }

    squaresCorners.clear();
    squaresCorners.reserve((horLines.size() - 1) * (vertLines.size() - 1));
    for (size_t i = 0; i < intersections.size() - 1; ++i) {
        for (size_t j = 0; j < intersections[i].size() - 1; ++j) {
            std::vector<cv::Point2f> square(4);
            square[0] = intersections[i][j];
            square[1] = intersections[i][j + 1];
            square[2] = intersections[i + 1][j + 1];
            square[3] = intersections[i + 1][j];
            squaresCorners.push_back(square);
        }
    }
}

void BoardDetector::getExtendedPolygon(const cv::Mat &img,
                                       const std::vector<cv::Point2f> &square,
                                       const double &avgHorSpacing, const double &avgVertSpacing,
                                       std::vector<cv::Point2f> &extendedSquare) const {
    extendedSquare.clear();

    auto validateAndPush = [&img, &extendedSquare](float X, float Y) {
        X = std::max(static_cast<float>(0), X);
        Y = std::max(static_cast<float>(0), Y);
        X = std::min(X, static_cast<float>(img.cols));
        Y = std::min(Y, static_cast<float>(img.rows));
        extendedSquare.emplace_back(X, Y);
    };

    // Top Left
    float extX = square[0].x - avgHorSpacing * config.squareExpansionRatioX;
    float extY = square[0].y - avgVertSpacing * config.squareExpansionRatioY;
    validateAndPush(extX, extY);

    // Top Right
    extX = square[1].x + avgHorSpacing * config.squareExpansionRatioX;
    extY = square[1].y - avgVertSpacing * config.squareExpansionRatioY;
    validateAndPush(extX, extY);

    // Bottom Right
    extX = square[2].x + avgHorSpacing * config.squareExpansionRatioX;
    extY = square[2].y + avgVertSpacing * config.squareExpansionRatioY;
    validateAndPush(extX, extY);

    // Bottom Left
    extX = square[3].x - avgHorSpacing * config.squareExpansionRatioX;
    extY = square[3].y + avgVertSpacing * config.squareExpansionRatioY;
    validateAndPush(extX, extY);
}

void BoardDetector::drawGrid(cv::Mat &img, const std::vector<std::vector<cv::Point2f> > &squaresCorners,
                             const std::vector<bool> *mask) {
    if (!mask) {
        const std::vector<bool> m(squaresCorners.size(), true);
        mask = &m;
    }

    for (size_t i = 0; i < squaresCorners.size(); ++i) {
        if ((*mask)[i]) {
            std::vector<cv::Point> pts;
            pts.reserve(squaresCorners[i].size());

            for (const auto &p: squaresCorners[i]) {
                pts.emplace_back(cvRound(p.x), cvRound(p.y));
            }

            cv::fillConvexPoly(img, pts, cv::Scalar(255, 0, 0), cv::LINE_AA);
            cv::polylines(img, pts, true, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
        }
    }
}

/**********************************************************************************/
/******************************** UNIFORMITY CHECK ********************************/
/**********************************************************************************/

void BoardDetector::getCandidatePieces(
    const cv::Mat &img, const cv::Mat &gray, std::vector<std::vector<cv::Point2f> > &squaresCorners,
    const double avgHorSpace, const double avgVertSpace,
    std::vector<bool> &emptySquares, Detection &output) const {
    cv::Mat edges;
    cv::Canny(gray, edges, config.cannyLT, config.cannyHT);
    for (size_t i = 0; i < squaresCorners.size(); ++i)
        emptySquares[i] = hasUniformFill(gray, edges, squaresCorners[i]);

    // Output set-up
    output.cellPositions.clear();
    output.img = img;
    for (const bool empty: emptySquares) output.cellPositions.push_back(!empty);

    // Update Detection vectors (containing occupied cells infos)
    for (size_t i = 0; i < emptySquares.size(); ++i) {
        if (!emptySquares[i]) {
            cv::Mat mask = cv::Mat::zeros(gray.size(), CV_8UC1);
            cv::fillConvexPoly(mask, GeometryUtils::point2fToPoint(squaresCorners[i]), cv::Scalar(255));
            output.cellMask.push_back(mask);

            cv::Mat extendedMask = cv::Mat::zeros(gray.size(), CV_8UC1);
            std::vector<cv::Point2f> extendedSquare = {};
            getExtendedPolygon(gray, squaresCorners[i], avgHorSpace, avgVertSpace, extendedSquare);
            cv::fillConvexPoly(extendedMask, GeometryUtils::point2fToPoint(extendedSquare), cv::Scalar(255));
            output.extendedCellMask.push_back(extendedMask);

            output.cellCorners.push_back(squaresCorners[i]);
        }
    }
}

bool BoardDetector::hasUniformFill(const cv::Mat &img, const cv::Mat &edges,
                                   const std::vector<cv::Point2f> &square) const {
    // Outer mask
    cv::Mat outerMask = cv::Mat::zeros(img.size(), CV_8UC1);
    cv::fillConvexPoly(
        outerMask,
        GeometryUtils::point2fToPoint(GeometryUtils::shrinkPolygon(square, 2)),
        cv::Scalar(255));

    // inner mask
    cv::Mat coreMask = cv::Mat::zeros(img.size(), CV_8UC1);
    cv::fillConvexPoly(
        coreMask,
        GeometryUtils::point2fToPoint(GeometryUtils::shrinkPolygon(square, 8)),
        cv::Scalar(255));

    // count # of edge pixels
    cv::Mat outerEdges, coreEdges;
    edges.copyTo(outerEdges, outerMask);
    edges.copyTo(coreEdges, coreMask);
    const int outerEdgePixels = cv::countNonZero(outerEdges);
    const int coreEdgePixels = cv::countNonZero(coreEdges);

    // if the edges are only on the outer mask, it is probably the portion of close square's piece exceeding its borders
    const bool adjacentPiece = (outerEdgePixels > config.minOuterEdgePixelsAdjacent) && (
                                   coreEdgePixels <= config.maxCoreEdgePixelsEmpty);
    const bool fewEdges = (outerEdgePixels <= 10) || adjacentPiece;

    cv::Scalar mean, stddev;
    cv::meanStdDev(img, mean, stddev, coreMask);
    const bool smallStddev = stddev[0] < config.maxEmptySquareStdDev;

    return smallStddev && fewEdges;
}

bool BoardDetector::estimateTopLeftColor(const cv::Mat &gray,
                                         const std::vector<std::vector<cv::Point2f> > &squaresCorners,
                                         const std::vector<bool> &emptySquares) {
    // if we couldn't detect a 64x64 chessboard, default to white topLeft color
    if (squaresCorners.size() != 64 || std::count(emptySquares.begin(), emptySquares.end(), true) < 5)
        return true;

    // Compute the average gray level to act as a threshold for light/dark cells
    cv::Mat boardMask = cv::Mat::zeros(gray.size(), CV_8UC1);
    for (const auto &square: squaresCorners) {
        cv::fillConvexPoly(boardMask, GeometryUtils::point2fToPoint(square),
                           cv::Scalar(255));
    }
    const double threshold = cv::mean(gray, boardMask)[0];

    // Compute the indices of 5 random empty cells
    std::vector<int> emptyIndices;
    for (size_t i = 0; i < emptySquares.size(); ++i) {
        if (emptySquares[i]) emptyIndices.push_back(static_cast<int>(i));
    }
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(emptyIndices.begin(), emptyIndices.end(), g);

    // We look at 5 random empty squares in the chessboard
    // We compute each cell's color, and determine the top left cell color that is coherent with the current cell
    // Each cell votes for a top left cell color
    int votesForLight = 0;
    int votesForDark = 0;
    const int cellsToCheck = std::min(5, static_cast<int>(emptyIndices.size()));
    int checkedCells = 0;

    // Iterate on the 5 random cells
    for (int k = 0; k < cellsToCheck; ++k) {
        // Cell true index on the chessboard
        const int i = emptyIndices[k];

        // Cell mask
        cv::Mat cellMask = cv::Mat::zeros(gray.size(), CV_8UC1);
        std::vector<cv::Point2f> reducedSquare = GeometryUtils::shrinkPolygon(squaresCorners[i], 4);
        cv::fillConvexPoly(cellMask, GeometryUtils::point2fToPoint(reducedSquare), cv::Scalar(255));

        // Compute the mean gray level
        const double cellMean = cv::mean(gray, cellMask)[0];
        const bool isCellLight = (cellMean > threshold);

        // Compute the top left cell color based on the current cell's color
        const int row = i / 8;
        const int col = i % 8;
        const bool isSameColorAsTopLeft = ((row + col) % 2 == 0);
        const bool estimatedTopLeftIsLight = isSameColorAsTopLeft ? isCellLight : !isCellLight;

        // Update votes
        if (estimatedTopLeftIsLight) {
            votesForLight++;
        } else {
            votesForDark++;
        }
    }

    return (votesForLight > votesForDark);
}

/**********************************************************************************/
/**************************** DEBUG INTERMEDIATEPRINTS ****************************/
/**********************************************************************************/

void BoardDetector::DEBUG_print1(const cv::Mat &gray,
                                 const cv::Ptr<cv::LineSegmentDetector> &lsd,
                                 const std::vector<cv::Vec4f> &allSegments,
                                 const std::vector<cv::Vec4f> &filteredSegments) {
    cv::Mat debugImg;
    cv::cvtColor(gray, debugImg, cv::COLOR_GRAY2BGR);
    lsd->compareSegments(debugImg.size(), allSegments, filteredSegments, debugImg);
    const std::string name = "debug_img" + std::to_string(DEBUG_window_index) + "_clusteredLSD";
    saveDebugImage(debugImg, name);
}

void BoardDetector::DEBUG_print2(const cv::Mat &gray, const std::vector<CollinearSegments> &lineClusters) {
    cv::Mat debugImg1;
    cv::cvtColor(gray, debugImg1, cv::COLOR_GRAY2BGR);
    for (size_t i = 0; i < lineClusters.size(); ++i) {
        // std::cout << "Line " << i << ";" << std::endl;
        // std::cout << "  Coverage: " << lineClusters[i].projection.sumIntervals() / std::max(img.rows, img.cols) <<
        //         std::endl;
        // std::cout << "  Dir: " << lineClusters[i].dir << std::endl;
        // std::cout << "  Avg gap: " << lineClusters[i].projection.avgGap() / std::max(img.rows, img.cols) <<
        //         std::endl;
        // std::cout << "  Intervals: [";
        // for (auto &s: lineClusters[i].projection.intervals)
        //     std::cout << "(" << s.first << ", " << s.second << "), ";
        // std::cout << "]" << std::endl;
        // std::cout << "  Gaps: [";
        // for (auto &p: lineClusters[i].projection.gaps) std::cout << p << ", ";
        // std::cout << "]" << std::endl;
        // Draw
        GeometryUtils::drawLine(debugImg1, lineClusters[i].line, cv::Scalar(0, 255, 0));
        for (auto &segment: lineClusters[i].segments) {
            GeometryUtils::drawSegment(debugImg1, segment, cv::Scalar(0, 0, 255));
        }
    }
    const std::string name1 = "debug_img" + std::to_string(DEBUG_window_index) + "_geometricFiltered";
    saveDebugImage(debugImg1, name1);
}

void BoardDetector::DEBUG_print3(
    const cv::Mat &gray, const std::vector<CollinearSegments> &horClusters,
    const std::vector<CollinearSegments> &vertClusters, const double thickness, const std::string &msg
) {
    cv::Mat debugImg3;
    cv::cvtColor(gray, debugImg3, cv::COLOR_GRAY2BGR);
    for (auto &horCluster: horClusters) {
        GeometryUtils::drawLine(debugImg3, horCluster.line, cv::Scalar(0, 255, 0));
        for (auto &segment: horCluster.segments) {
            GeometryUtils::drawSegment(debugImg3, segment, cv::Scalar(179, 65, 28), thickness);
        }
    }
    for (auto &vertCluster: vertClusters) {
        GeometryUtils::drawLine(debugImg3, vertCluster.line, cv::Scalar(0, 255, 0));
        for (auto &segment: vertCluster.segments) {
            GeometryUtils::drawSegment(debugImg3, segment, cv::Scalar(0, 0, 255), thickness);
        }
    }
    const std::string name3 = "debug_img" + std::to_string(DEBUG_window_index) + "_" + msg;
    saveDebugImage(debugImg3, name3);
}

void BoardDetector::DEBUG_print4(const cv::Mat &gray, const std::vector<std::vector<cv::Point2f> > &squaresCorners,
                                 const std::vector<bool> &emptySquares) {
    cv::Mat debugImg4;
    cv::cvtColor(gray, debugImg4, cv::COLOR_GRAY2BGR);
    drawGrid(debugImg4, squaresCorners, &emptySquares);
    const std::string name4 = "debug_img" + std::to_string(DEBUG_window_index) + "_emptyCellsGridView";
    saveDebugImage(debugImg4, name4);
}
