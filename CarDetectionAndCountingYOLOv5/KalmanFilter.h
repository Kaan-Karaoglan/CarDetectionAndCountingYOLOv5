#ifndef KALMANFILTER_H
#define KALMANFILTER_H

#include <opencv2/opencv.hpp>

class KalmanFilter {
public:
    KalmanFilter();
    void init(const cv::Point2f& initial_position);
    cv::Point2f predict();
    void update(const cv::Point2f& measured_position);

private:
    cv::KalmanFilter kf;
    cv::Mat state;       // [x, y, vx, vy] - Durum vektörü
    cv::Mat measurement; // [x, y] - Ölçüm vektörü
};

#endif // KALMANFILTER_H
