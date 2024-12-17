#include "KalmanFilter.h"

KalmanFilter::KalmanFilter() {
    // Kalman filtresi 4 durumlu (x, y, vx, vy) ve 2 ölçümlü (x, y) olarak baþlatýlýr
    kf.init(4, 2, 0);

    // Durum geçiþ matrisi (A)
    kf.transitionMatrix = (cv::Mat_<float>(4, 4) <<
        1, 0, 1, 0,
        0, 1, 0, 1,
        0, 0, 1, 0,
        0, 0, 0, 1);

    // Ölçüm matrisi (H)
    kf.measurementMatrix = (cv::Mat_<float>(2, 4) <<
        1, 0, 0, 0,
        0, 1, 0, 0);

    // Süreç gürültüsü kovaryans matrisi (Q)
    kf.processNoiseCov = (cv::Mat_<float>(4, 4) <<
        1, 0, 0.1, 0,
        0, 1, 0, 0.1,
        0.1, 0, 1, 0,
        0, 0.1, 0, 1);

    // Ölçüm gürültüsü kovaryans matrisi (R)
    kf.measurementNoiseCov = (cv::Mat_<float>(2, 2) <<
        1, 0,
        0, 1);

    // Baþlangýç durum vektörü
    state = cv::Mat::zeros(4, 1, CV_32F);

    // Ölçüm vektörü
    measurement = cv::Mat::zeros(2, 1, CV_32F);
}

void KalmanFilter::init(const cv::Point2f& initial_position) {
    // Baþlangýç pozisyonunu ve hýzýný belirle
    state.at<float>(0) = initial_position.x;
    state.at<float>(1) = initial_position.y;
    state.at<float>(2) = 0.0f; // vx
    state.at<float>(3) = 0.0f; // vy

    // Kalman filtresine baþlangýç durumunu atayýn
    kf.statePost = state;
}

cv::Point2f KalmanFilter::predict() {
    // Gelecekteki pozisyonu tahmin et
    cv::Mat prediction = kf.predict();
    return cv::Point2f(prediction.at<float>(0), prediction.at<float>(1));
}

void KalmanFilter::update(const cv::Point2f& measured_position) {
    // Ölçüm vektörünü güncelle
    measurement.at<float>(0) = measured_position.x;
    measurement.at<float>(1) = measured_position.y;

    // Ölçüm ve tahmin deðerlerini kullanarak filtreyi güncelle
    kf.correct(measurement);
}
