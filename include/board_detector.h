// Author: Leonardo Gusson, 2157199

#ifndef CV_FINALPROJECT_BOARD_DETECTOR_H
#define CV_FINALPROJECT_BOARD_DETECTOR_H

#include <opencv2/opencv.hpp>
#include "geometry_config.h"
#include "geometry_utils.h"

/**
 * @brief Represents the result of the chessboard and pieces detection.
 * Contains the original image, grid geometry, and masks for the occupied cells.
 */
struct Detection {
    cv::Mat img;

    // True if the top-left square of the board is estimated to be light (white), false if dark (black)
    bool isFirstWhite;
    // Boolean vector (typically 64 elements) indicating which squares contain a piece (true = occupied, false = empty)
    std::vector<bool> cellPositions;

    // Vector of binary masks (exact square size) for each occupied cell
    std::vector<cv::Mat> cellMask;

    // Vector of expanded binary masks for each occupied cell
    std::vector<cv::Mat> extendedCellMask;

    // Vector containing the 4 corners (cv::Point2f) for each chessboard square
    std::vector<std::vector<cv::Point2f> > cellCorners;
};

class BoardDetector {
public:
    /**
     * @brief Default constructor. Initializes the configuration.
     */
    BoardDetector() : config{} {
    }

    /**
     * @brief Detects the board and extracts occupied squares.
     * @param img The input BGR image.
     * @return A Detection object containing the chessboard grid info, cell masks, and pieces positions.
     */
    Detection detect(const cv::Mat &img) const;

private:
    /***********************************************************************************/
    /******************************* GEOMETRIC FILTERING *******************************/
    /***********************************************************************************/

    /**
     * @brief Groups smaller collinear line segments into larger clusters.
     * @param segments Vector of raw line segments detected by LSD.
     * @return Vector of grouped collinear segments.
     */
    std::vector<CollinearSegments> mergeAlignedSegments(const std::vector<cv::Vec4f> &segments) const;

    /**
     * @brief Merges overlapping collinear clusters into single representations.
     * @param clusters Reference to the vector of line clusters to be simplified.
     */
    void removeCollinearClusters(std::vector<CollinearSegments> &clusters) const;

    /**
     * @brief Discards line clusters with poor coverage or excessively large gaps.
     * @param clusters Reference to the vector of line clusters.
     * @param imgMaxDim The maximum dimension (width or height) of the image.
     */
    void geometricFilter(std::vector<CollinearSegments> &clusters, int imgMaxDim) const;

    /**********************************************************************************/
    /******************************* ANGULAR FILTERING ********************************/
    /**********************************************************************************/

    /**
     * @brief Splits lines into horizontal and vertical sets using K-means clustering on their angles.
     * @param clusters Input vector of line clusters.
     * @param horClusters Output vector for horizontal lines.
     * @param vertClusters Output vector for vertical lines.
     */
    void angularClustering(std::vector<CollinearSegments> &clusters,
                           std::vector<CollinearSegments> &horClusters,
                           std::vector<CollinearSegments> &vertClusters) const;

    /**
     * @brief Computes and sets the absolute angle [0, pi] for each cluster.
     * @param clusters Vector of line clusters.
     */
    static void computeClustersAngle(std::vector<CollinearSegments> &clusters);

    /**
     * @brief Filters out lines that deviate too much from the dominant group angle or vanishing point.
     * @param lines Reference to the vector of horizontal or vertical lines to filter.
     */
    void angularFilter(std::vector<CollinearSegments> &lines) const;

    /**********************************************************************************/
    /******************************* SPATIAL FILTERING ********************************/
    /**********************************************************************************/

    /**
     * @brief Sorts horizontal lines top-to-bottom and vertical lines left-to-right.
     * @param horClusters Vector of horizontal lines.
     * @param vertClusters Vector of vertical lines.
     * @param imgSize The dimensions of the image.
     * @param horSpacing Output vector storing the vertical positions of horizontal lines.
     * @param vertSpacing Output vector storing the horizontal positions of vertical lines.
     */
    static void sortLinesSpatially(std::vector<CollinearSegments> &horClusters,
                                   std::vector<CollinearSegments> &vertClusters,
                                   const cv::Size &imgSize, std::vector<double> &horSpacing,
                                   std::vector<double> &vertSpacing);

    /**
     * @brief Filters lines based on regular spacing to identify the actual chessboard grid.
     * @param horClusters Reference to horizontal lines.
     * @param vertClusters Reference to vertical lines.
     * @param horSpacing Spatial positions of horizontal lines.
     * @param vertSpacing Spatial positions of vertical lines.
     * @param imgSize Image dimensions.
     * @param avgHorSpacing Output average distance between horizontal grid lines.
     * @param avgVertSpacing Output average distance between vertical grid lines.
     * @return True if a valid grid is found, false otherwise.
     */
    bool spatialGridFilter(std::vector<CollinearSegments> &horClusters,
                           std::vector<CollinearSegments> &vertClusters,
                           std::vector<double> &horSpacing, std::vector<double> &vertSpacing,
                           const cv::Size &imgSize, double &avgHorSpacing, double &avgVertSpacing) const;

    /**
     * @brief Finds a starting sequence of consecutive, evenly spaced lines to act as a grid seed.
     * @param gaps Distances between adjacent lines.
     * @param seeds Output vector with the indices of the seed lines.
     * @param avgGap Output average gap distance within the seed.
     * @return True if a valid seed is found.
     */
    bool getSeeds(const std::vector<double> &gaps, std::vector<int> &seeds, double &avgGap) const;

    /**
     * @brief Returns the longest sequence of consecutive integers from a vector.
     * @param v Input vector of integers.
     * @return Vector containing the longest consecutive sequence.
     */
    static std::vector<int> getConsecutive(const std::vector<int> &v);

    /**
     * @brief Expands the grid starting from the seed lines based on the estimated regular spacing.
     * @param spacing Vector of line positions.
     * @param seedGapIndexes Indices of the previously found seed gaps.
     * @param imgExtension Image dimension along the axis of expansion.
     * @param outFilter Output boolean mask indicating which lines belong to the grid.
     */
    void expandSeeds(const std::vector<double> &spacing, const std::vector<int> &seedGapIndexes,
                     int imgExtension, std::vector<bool> &outFilter) const;

    /**********************************************************************************/
    /********************************* GRID EXTRACTION ********************************/
    /**********************************************************************************/

    /**
     * @brief Computes line intersections to generate the 4 corners for each board square.
     * @param horLines Filtered horizontal grid lines.
     * @param vertLines Filtered vertical grid lines.
     * @param squaresCorners Output matrix of squares, each defined by 4 corner points.
     */
    static void extractGrid(const std::vector<CollinearSegments> &horLines,
                            const std::vector<CollinearSegments> &vertLines,
                            std::vector<std::vector<cv::Point2f> > &squaresCorners);

    /**
     * @brief Computes an enlarged version of a convex polygon to capture piece overflows.
     * @param img The input image used for bounding checks.
     * @param square The 4 corners of the standard square.
     * @param avgHorSpacing Average horizontal cell width.
     * @param avgVertSpacing Average vertical cell height.
     * @param extendedSquare Output corners of the enlarged square.
     */
    void getExtendedPolygon(const cv::Mat &img,
                            const std::vector<cv::Point2f> &square,
                            const double &avgHorSpacing, const double &avgVertSpacing,
                            std::vector<cv::Point2f> &extendedSquare) const;

    /**
     * @brief Draws the detected grid and squares on an image.
     * @param img The image to draw on.
     * @param squaresCorners The list of squares to draw.
     * @param mask Optional boolean mask to draw only specific squares (e.g., empty/occupied).
     */
    static void drawGrid(cv::Mat &img, const std::vector<std::vector<cv::Point2f> > &squaresCorners,
                         const std::vector<bool> *mask = nullptr);

    /**********************************************************************************/
    /******************************** UNIFORMITY CHECK ********************************/
    /**********************************************************************************/

    /**
     * @brief Evaluates all detected squares and outputs those containing pieces.
     * @param img Input BGR image.
     * @param gray Input grayscale image.
     * @param squaresCorners Vector of all board squares.
     * @param avgHorSpace Average cell width.
     * @param avgVertSpace Average cell height.
     * @param emptySquares Output boolean vector marking which squares are empty.
     * @param output Output Detection object representing the whole chessboard.
     */
    void getCandidatePieces(
        const cv::Mat &img, const cv::Mat &gray,
        std::vector<std::vector<cv::Point2f> > &squaresCorners,
        double avgHorSpace, double avgVertSpace, std::vector<bool> &emptySquares,
        Detection &output) const;

    /**
     * @brief Determines if a square is empty by analyzing its texture variance and inner edge count.
     * @param img Grayscale image.
     * @param edges Canny edge image.
     * @param square The 4 corners of the square to check.
     * @return True if the square is empty (uniform fill), false if it likely contains a piece.
     */
    bool hasUniformFill(const cv::Mat &img, const cv::Mat &edges, const std::vector<cv::Point2f> &square) const;

    /**
     * @brief Estimates the color (light or dark) of the top-left square of the board.
     * @param gray The input grayscale image.
     * @param squaresCorners Vector of all 64 board squares.
     * @param emptySquares Boolean vector indicating which squares are empty.
     * @return True if the top-left square is estimated to be light (white), false if dark (black).
     */
    static bool estimateTopLeftColor(const cv::Mat &gray,
                                     const std::vector<std::vector<cv::Point2f> > &squaresCorners,
                                     const std::vector<bool> &emptySquares);

    /**********************************************************************************/
    /************************************* DEBUG **************************************/
    /**********************************************************************************/

    /**
     * @brief Debug display showing initial LSD segments vs length-filtered segments.
     */
    static void DEBUG_print1(const cv::Mat &gray,
                             const cv::Ptr<cv::LineSegmentDetector> &lsd,
                             const std::vector<cv::Vec4f> &allSegments,
                             const std::vector<cv::Vec4f> &filteredSegments);

    /**
     * @brief Debug display showing the geometric clustered lines.
     */
    static void DEBUG_print2(const cv::Mat &gray, const std::vector<CollinearSegments> &lineClusters);

    /**
     * @brief Debug display showing the angular or spatial filtered horizontal and vertical clusters.
     */
    static void DEBUG_print3(const cv::Mat &gray,
                             const std::vector<CollinearSegments> &horClusters,
                             const std::vector<CollinearSegments> &vertClusters,
                             double thickness, const std::string &msg);

    /**
     * @brief Debug display showing the final extracted grid and empty/occupied square distinction.
     */
    static void DEBUG_print4(const cv::Mat &gray, const std::vector<std::vector<cv::Point2f> > &squaresCorners,
                             const std::vector<bool> &emptySquares);

    const Config config;
    static int DEBUG_window_index;
};

#endif //CV_FINALPROJECT_BOARD_DETECTOR_H
