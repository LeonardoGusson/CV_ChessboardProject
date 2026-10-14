// Author: Leonardo Gusson, 2157199

#ifndef CV_FINALPROJECT_GEOMETRY_CONFIG_H
#define CV_FINALPROJECT_GEOMETRY_CONFIG_H

struct Config {
    // First segments clustering (merge)
    double minSegmentLengthRatio = 0.025;
    double parallelismAngleThreshold = 0.0785398; // 4.5°
    double collinearityDistThreshold = 3; // px

    // Removal of collinear clusters
    double mergeCollinearAngleThreshold = 0.0523599; // 3°
    int mergeCollinearDistThreshold = 3;

    // Clusters geometric filtering
    int minSegmentsPerCluster = 3;
    double minLineCoverageRatio = 0.175;
    double highLineCoverageRatio = 0.4;
    double maxGapRatio = 0.20;
    double maxAvgGapRatio = 0.10;

    // angular filtering
    double maxClusterAngleDeviation = 0.349066; // 20°
    double maxParallelStdDev = 0.0349066; // 2°
    double sigmaFilter = 2;
    double vanishingPointDistThreshold = 10;

    // spatial filtering
    double gapSimilarityThreshold = 0.2; // % over the gap extension
    int minGapsCount = 2;
    double gapExpansionToleranceThreshold = 0.15; // % over the last gap

    // Polygon extension
    double squareExpansionRatioX = 0.5;
    double squareExpansionRatioY = 0.5;

    // Chessboard square content
    double cannyLT = 80;
    double cannyHT = 200;
    double maxEmptySquareStdDev = 30;
    int maxCoreEdgePixelsEmpty = 3;
    int minOuterEdgePixelsAdjacent = 10;

    // Debug
    bool debug1 = false;
    bool debug2 = false;
    bool debug3a = false;
    bool debug3b = false;
    bool debug4 = false;
};


#endif //CV_FINALPROJECT_GEOMETRY_CONFIG_H
