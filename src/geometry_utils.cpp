// Author: Leonardo Gusson, 2157199

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "geometry_utils.h"

// ======================================================================== //
// =========================== PROJECTION CLASS =========================== //
// ======================================================================== //

double Projection::sumIntervals() const {
    double sum = 0;
    for (const auto &interval: intervals) sum += std::abs(interval.second - interval.first);
    return sum;
}

double Projection::maxGap() const {
    if (gaps.empty()) return 0;
    return *std::max_element(gaps.begin(), gaps.end());
}

double Projection::avgGap() const {
    if (gaps.empty()) return 0;
    double sum = 0;
    for (const auto &gap: gaps) sum += gap;
    return sum / gaps.size();
}

// ======================================================================== //
// ======================= COLLINEAR SEGMENTS CLASS ======================= //
// ======================================================================== //

void CollinearSegments::addSegment(const cv::Vec4f &segment) {
    segments.push_back(segment);
    fitLine();
}

void CollinearSegments::fitLine() {
    std::vector<cv::Point2f> points;
    points.reserve(segments.size() * 2);
    for (const auto &s: segments) {
        points.emplace_back(s[0], s[1]); // Punto A
        points.emplace_back(s[2], s[3]); // Punto B
    }

    if (points.empty()) {
        line = cv::Vec3f(0, 0, 0);
        dir = cv::Vec2f(0, 0); // Importante azzerare anche dir
        return;
    }

    // Output line: (vx, vy, x0, y0)
    cv::Vec4f out_line;
    cv::fitLine(points, out_line, cv::DIST_L2, 0, 0.01, 0.01);

    const float vx = out_line[0];
    const float vy = out_line[1];
    const float x0 = out_line[2];
    const float y0 = out_line[3];

    line[0] = vy;
    line[1] = -vx;
    line[2] = vx * y0 - vy * x0;
    dir[0] = vx;
    dir[1] = vy;
}

void CollinearSegments::merge(CollinearSegments &c1, const CollinearSegments &c2) {
    c1.segments.insert(c1.segments.end(), c2.segments.begin(), c2.segments.end());
    c1.fitLine();
    c1.project();
}

void CollinearSegments::project() {
    if (segments.empty()) projection = Projection{{}, {}};

    // Initialize the vector of projected intervals
    std::vector<std::pair<double, double> > intervals;
    intervals.reserve(segments.size());

    // Populate the vector
    for (cv::Vec4f &segment: segments) {
        cv::Vec2f p1(segment[0], segment[1]);
        cv::Vec2f p2(segment[2], segment[3]);
        double p1_proj = p1.dot(dir);
        double p2_proj = p2.dot(dir);
        if (p1_proj > p2_proj) std::swap(p1_proj, p2_proj);
        intervals.emplace_back(p1_proj, p2_proj);
    }

    // Sort the intervals
    std::sort(intervals.begin(), intervals.end(),
              [](const std::pair<double, double> &a, const std::pair<double, double> &b) {
                  return a.first < b.first;
              });

    // Merge overlapping intervals & compute gaps
    std::vector<std::pair<double, double> > fin_intervals = {intervals.front()};
    std::vector<double> gaps = {};
    for (size_t i = 1; i < intervals.size(); ++i) {
        if (intervals[i].first < fin_intervals.back().second) {
            fin_intervals.back().second = std::max(fin_intervals.back().second, intervals[i].second);
        } else {
            gaps.push_back(std::abs(intervals[i].first - fin_intervals.back().second));
            fin_intervals.push_back(intervals[i]);
        }
    }

    projection = Projection{fin_intervals, gaps};
}

// ======================================================================== //
// ========================= GEOMETRY UTILS CLASS ========================= //
// ======================================================================== //

cv::Vec3f GeometryUtils::pointsToLineABC(const cv::Vec4f &l) {
    float a = l[3] - l[1];
    float b = l[0] - l[2];
    float c = (l[2] - l[0]) * l[1] - (l[3] - l[1]) * l[0];
    const float norm = std::hypot(a, b);
    if (norm > 1e-6f) {
        a /= norm;
        b /= norm;
        c /= norm;
    }
    return {a, b, c};
}

float GeometryUtils::pointLineDist(const cv::Point2f &point, const cv::Vec3f &line) {
    // |a xP + b yP + c| / sqrt( a^2 + b^2 )
    const float num = std::fabs(line[0] * point.x + line[1] * point.y + line[2]);
    const float den = sqrt(line[0] * line[0] + line[1] * line[1]);
    return num / den;
}

double GeometryUtils::angleBetweenLines(const cv::Vec3f &line1, const cv::Vec3f &line2) {
    cv::Vec2f n1(line1[0], line1[1]);
    cv::Vec2f n2(line2[0], line2[1]);

    // dot = |n1||n2|cos(theta)
    const float dot = n1.dot(n2);
    // det = |n1||n2|sin(theta)
    const float det = n1[0] * n2[1] - n1[1] * n2[0];

    // det/dot = tan(theta)
    return std::atan2(std::abs(det), std::abs(dot));
}

bool GeometryUtils::areParallel(const cv::Vec3f &line1, const cv::Vec3f &line2, const double parallelismThreshold) {
    return angleBetweenLines(line1, line2) < parallelismThreshold;
}

bool GeometryUtils::areCollinear(const cv::Vec3f &line, const cv::Vec4f &seg, const double angleThreshold,
                                 const double distThreshold) {
    // Check parallelism
    const cv::Vec3f line2 = pointsToLineABC(seg);
    if (!areParallel(line, line2, angleThreshold)) return false;

    // Check collinearity
    const cv::Point2f p1(seg[0], seg[1]);
    const cv::Point2f p2(seg[2], seg[3]);
    if (pointLineDist(p1, line) < distThreshold && pointLineDist(p2, line) < distThreshold)
        return true;

    return false;
}

cv::Point2f GeometryUtils::findIntersection(const cv::Vec3f &line1, const cv::Vec3f &line2) {
    {
        cv::Vec3f p = line1.cross(line2);
        if (std::abs(p[2]) < 1e-6) {
            return {-1, -1};
        }

        return {p[0] / p[2], p[1] / p[2]};
    }
}

std::vector<cv::Point2f> GeometryUtils::shrinkPolygon(const std::vector<cv::Point2f> &p, float padding) {
    std::vector<cv::Point2f> points;
    points.reserve(p.size());

    if (p.size() != 4) return {};

    points.emplace_back(p[0].x + padding, p[0].y + padding);
    points.emplace_back(p[1].x - padding, p[1].y + padding);
    points.emplace_back(p[2].x - padding, p[2].y - padding);
    points.emplace_back(p[3].x + padding, p[3].y - padding);

    return points;
}


void GeometryUtils::drawLine(cv::Mat &image, const cv::Vec3f &line, const cv::Scalar &color, const double thickness) {
    const float a = line[0], b = line[1], c = line[2];

    std::vector<cv::Point2f> pts;
    if (b == 0) {
        // vertical line
        pts.emplace_back(-c / a, 0);
        pts.emplace_back(-c / a, image.size().height - 1);
    } else if (a == 0) {
        // horizontal line
        pts.emplace_back(0, -c / b);
        pts.emplace_back(image.size().width - 1, -c / b);
    } else {
        float yRight = (-a * (image.size().width - 1) - c) / b; // y = -ax-c / b
        float xBottom = (-b * (image.size().height - 1) - c) / a; // x = -by-c / a

        if (-c / b >= 0 && -c / b < image.size().height) pts.emplace_back(0, -c / b);
        if (-c / a >= 0 && -c / a < image.size().width) pts.emplace_back(-c / a, 0);
        if (yRight >= 0 && yRight < image.size().height) pts.emplace_back(image.size().width - 1, yRight);
        if (xBottom >= 0 && xBottom < image.size().width) pts.emplace_back(xBottom, image.size().height - 1);
    }
    if (pts.size() >= 2)
        cv::line(image, pts[0], pts[1], color, thickness);
}

void GeometryUtils::drawSegment(cv::Mat &image, const cv::Vec4f &segment, const cv::Scalar &color, double thickness) {
    const cv::Point pt1(cvRound(segment[0]), cvRound(segment[1]));
    const cv::Point pt2(cvRound(segment[2]), cvRound(segment[3]));
    cv::line(image, pt1, pt2, color, thickness);
}

cv::Point GeometryUtils::point2fToPoint(const cv::Point2f &p) {
    return {cvRound(p.x), cvRound(p.y)};
}

std::vector<cv::Point> GeometryUtils::point2fToPoint(const std::vector<cv::Point2f> &v) {
    std::vector<cv::Point> points;
    points.reserve(v.size());
    for (const auto &p: v) {
        points.emplace_back(point2fToPoint(p));
    }
    return points;
}
