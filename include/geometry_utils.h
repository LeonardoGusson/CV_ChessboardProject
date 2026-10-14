// Author: Leonardo Gusson 2157199

#ifndef CV_FINALPROJECT_GEOMETRY_UTILS_H
#define CV_FINALPROJECT_GEOMETRY_UTILS_H

#include <opencv2/core.hpp>
#include "geometry_config.h"

/**
 * @brief Handles the 1D projection of line segments onto their main reference line.
 */
class Projection {
public:
    /**
     * @brief Constructs a Projection object.
     * @param intervals Vector of start/end points of projected segments.
     * @param gaps Vector of distances between consecutive intervals.
     */
    Projection(const std::vector<std::pair<double, double> > &intervals, const std::vector<double> &gaps)
        : intervals(intervals), gaps(gaps) {
    }

    /**
     * @brief Calculates the total length of all projected intervals.
     * @return The sum of the lengths of all intervals.
     */
    double sumIntervals() const;

    /**
     * @brief Finds the largest gap between consecutive intervals.
     * @return The maximum gap size.
     */
    double maxGap() const;

    /**
     * @brief Computes the average gap size between intervals.
     * @return The average gap size.
     */
    double avgGap() const;

    std::vector<std::pair<double, double> > intervals; // Projected segments as [start, end] pairs
    std::vector<double> gaps; // Empty spaces between projected segments
};

/**
 * @brief Represents a group of line segments that share the same mathematical line.
 */
class CollinearSegments {
public:
    /**
     * @brief Constructor for multiple segments.
     * @param line Line coefficients (a, b, c).
     * @param dir Direction vector.
     * @param segments Vector of line segments.
     */
    CollinearSegments(const cv::Vec3f line, const cv::Vec2f dir, const std::vector<cv::Vec4f> segments)
        : line(line), dir(dir), segments(segments) {
    }

    /**
     * @brief Constructor for a single segment.
     * @param line Line coefficients (a, b, c).
     * @param dir Direction vector.
     * @param segment A single line segment.
     */
    CollinearSegments(const cv::Vec3f line, const cv::Vec2f dir, const cv::Vec4f segment)
        : line(line),
          dir(dir),
          segments(std::vector<cv::Vec4f>{segment}) {
    }

    /**
     * @brief Adds a new segment to the cluster and updates the line fit.
     * @param segment The segment to add.
     */
    void addSegment(const cv::Vec4f &segment);

    /**
     * @brief Recalculates the main line using least squares on all segments.
     */
    void fitLine();

    /**
     * @brief Merges two collinear segment clusters into one.
     * @param c1 The target cluster (modified).
     * @param c2 The source cluster to merge into c1.
     */
    static void merge(CollinearSegments &c1, const CollinearSegments &c2);

    /**
     * @brief Computes the 1D projection of all segments onto the reference line.
     */
    void project();

    cv::Vec3f line; // Line coefficients (a, b, c) for the equation ax + by + c = 0
    cv::Vec2f dir; // Direction vector (vx, vy) of the line
    double angle = 0; // Absolute angle of the line, used for clustering
    std::vector<cv::Vec4f> segments; // The raw line segments forming this cluster
    Projection projection = Projection{{}, {}}; // 1D projection data of the segments
};

class GeometryUtils {
public:
    GeometryUtils() : config{} {
    }

    /**
     * @brief Converts a 2-point segment into the ax + by + c = 0 line representation.
     * @param l Segment defined by 2 points (x1, y1, x2, y2).
     * @return Normalized line coefficients (a, b, c).
     */
    static cv::Vec3f pointsToLineABC(const cv::Vec4f &l);

    /**
     * @brief Calculates the perpendicular Euclidean distance from a point to a line.
     * @param point The 2D point.
     * @param line The line coefficients (a, b, c).
     * @return The shortest distance to the line.
     */
    static float pointLineDist(const cv::Point2f &point, const cv::Vec3f &line);

    /**
     * @brief Computes the angle in radians between two lines.
     * @param line1 First line (a, b, c).
     * @param line2 Second line (a, b, c).
     * @return The angle between the lines in radians.
     */
    static double angleBetweenLines(const cv::Vec3f &line1, const cv::Vec3f &line2);

    /**
     * @brief Checks if two lines are parallel within a given angular threshold.
     * @param line1 First line.
     * @param line2 Second line.
     * @param parallelismThreshold Maximum allowed angle difference in radians.
     * @return True if the lines are considered parallel.
     */
    static bool areParallel(const cv::Vec3f &line1, const cv::Vec3f &line2, double parallelismThreshold);

    /**
     * @brief Checks if a segment is collinear to a line based on angle and distance thresholds.
     * @param line Reference line (a, b, c).
     * @param seg Line segment to test.
     * @param angleThreshold Maximum allowed angle difference.
     * @param distThreshold Maximum allowed perpendicular distance from segment endpoints to the line.
     * @return True if the segment is collinear to the line.
     */
    static bool areCollinear(const cv::Vec3f &line, const cv::Vec4f &seg, double angleThreshold,
                             double distThreshold);

    /**
     * @brief Finds the 2D intersection point of two lines.
     * @param line1 First line.
     * @param line2 Second line.
     * @return The intersection point, or (-1, -1) if lines are parallel.
     */
    static cv::Point2f findIntersection(const cv::Vec3f &line1, const cv::Vec3f &line2);

    /**
     * @brief Shrinks a 4-point polygon (e.g., a square) inwards by a constant padding.
     * @param p Vector containing exactly 4 corner points.
     * @param padding Amount of pixels to shrink the polygon inwards.
     * @return The shrunken polygon points.
     */
    static std::vector<cv::Point2f> shrinkPolygon(const std::vector<cv::Point2f> &p, float padding);

    /** * @brief Draws an infinite line across an entire image.
     * @param image The image to draw on.
     * @param line Line coefficients (a, b, c).
     * @param color Line color.
     * @param thickness Line thickness.
     */
    static void drawLine(cv::Mat &image, const cv::Vec3f &line, const cv::Scalar &color, double thickness = 1.0);

    /**
     * @brief Draws a specific finite line segment on an image.
     * @param image The image to draw on.
     * @param segment Segment endpoints (x1, y1, x2, y2).
     * @param color Line color.
     * @param thickness Line thickness.
     */
    static void drawSegment(cv::Mat &image, const cv::Vec4f &segment, const cv::Scalar &color, double thickness = 1.0);

    /**
     * @brief Converts a floating-point 2D point to an integer 2D point.
     * @param p The floating-point point.
     * @return The rounded integer point.
     */
    static cv::Point point2fToPoint(const cv::Point2f &p);

    /**
     * @brief Converts a vector of floating-point 2D points to integer 2D points.
     * @param v The vector of floating-point points.
     * @return The vector of rounded integer points.
     */
    static std::vector<cv::Point> point2fToPoint(const std::vector<cv::Point2f> &v);

private:
    const Config config;
};

#endif //CV_FINALPROJECT_GEOMETRY_UTILS_H
