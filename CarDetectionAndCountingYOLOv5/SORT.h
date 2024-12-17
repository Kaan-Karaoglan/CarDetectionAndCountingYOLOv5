#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <deque>
#include <unordered_map>
#include <algorithm>

class SORT {
public:
    SORT(float iou_threshold = 0.3);
    std::vector<std::vector<int>> update(const std::vector<cv::Rect>& detections); // Deðiþiklik burada    
    // void update(const std::vector<cv::Rect>& detections);
    const std::vector<int>& getTrackedObjectIDs() const;

private:
    float iou_threshold;
    int next_id;
    std::unordered_map<int, cv::Rect> tracked_objects;
    std::unordered_map<int, cv::Point> object_centers;
    std::unordered_map<int, bool> object_crossed;
    std::vector<int> tracked_ids;

    float computeIoU(const cv::Rect& box1, const cv::Rect& box2);
    void associateDetectionsToTrackedObjects(const std::vector<cv::Rect>& detections);
    void updateObjectTracking(const std::vector<cv::Rect>& detections);
};
