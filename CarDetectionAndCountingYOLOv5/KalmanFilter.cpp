#include "KalmanFilter.h"

KalmanFilter::KalmanFilter() {
    // Kalman filtresi 4 durumlu (x, y, vx, vy) ve 2 �l��ml� (x, y) olarak ba�lat�l�r
    kf.init(4, 2, 0);

    // Durum ge�i� matrisi (A)
    kf.transitionMatrix = (cv::Mat_<float>(4, 4) <<
        1, 0, 1, 0,
        0, 1, 0, 1,
        0, 0, 1, 0,
        0, 0, 0, 1);

    // �l��m matrisi (H)
    kf.measurementMatrix = (cv::Mat_<float>(2, 4) <<
        1, 0, 0, 0,
        0, 1, 0, 0);

    // S�re� g�r�lt�s� kovaryans matrisi (Q)
    kf.processNoiseCov = (cv::Mat_<float>(4, 4) <<
        1, 0, 0.1, 0,
        0, 1, 0, 0.1,
        0.1, 0, 1, 0,
        0, 0.1, 0, 1);

    // �l��m g�r�lt�s� kovaryans matrisi (R)
    kf.measurementNoiseCov = (cv::Mat_<float>(2, 2) <<
        1, 0,
        0, 1);

    // Ba�lang�� durum vekt�r�
    state = cv::Mat::zeros(4, 1, CV_32F);

    // �l��m vekt�r�
    measurement = cv::Mat::zeros(2, 1, CV_32F);
}

void KalmanFilter::init(const cv::Point2f& initial_position) {
    // Ba�lang�� pozisyonunu ve h�z�n� belirle
    state.at<float>(0) = initial_position.x;
    state.at<float>(1) = initial_position.y;
    state.at<float>(2) = 0.0f; // vx
    state.at<float>(3) = 0.0f; // vy

    // Kalman filtresine ba�lang�� durumunu atay�n
    kf.statePost = state;
}

cv::Point2f KalmanFilter::predict() {
    // Gelecekteki pozisyonu tahmin et
    cv::Mat prediction = kf.predict();
    return cv::Point2f(prediction.at<float>(0), prediction.at<float>(1));
}

void KalmanFilter::update(const cv::Point2f& measured_position) {
    // �l��m vekt�r�n� g�ncelle
    measurement.at<float>(0) = measured_position.x;
    measurement.at<float>(1) = measured_position.y;

    // �l��m ve tahmin de�erlerini kullanarak filtreyi g�ncelle
    kf.correct(measurement);
}
