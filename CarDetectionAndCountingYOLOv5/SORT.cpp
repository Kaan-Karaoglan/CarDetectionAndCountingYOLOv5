#include "SORT.h"
#include <algorithm>
#include <opencv2/opencv.hpp>

SORT::SORT(float iou_threshold)
    : iou_threshold(iou_threshold), next_id(0) {
}

float SORT::computeIoU(const cv::Rect& box1, const cv::Rect& box2) {
    int x1 = std::max(box1.x, box2.x);
    int y1 = std::max(box1.y, box2.y);
    int x2 = std::min(box1.x + box1.width, box2.x + box2.width);
    int y2 = std::min(box1.y + box1.height, box2.y + box2.height);

    int intersection_area = std::max(0, x2 - x1) * std::max(0, y2 - y1);
    int union_area = box1.area() + box2.area() - intersection_area;

    return float(intersection_area) / float(union_area);
}

void SORT::associateDetectionsToTrackedObjects(const std::vector<cv::Rect>& detections) {
    std::vector<int> unmatched_detections;
    std::vector<int> matched_detections;

    for (size_t i = 0; i < detections.size(); ++i) {
        bool matched = false;
        for (auto& tracked : tracked_objects) {
            if (computeIoU(detections[i], tracked.second) > iou_threshold) {
                matched = true;
                object_centers[tracked.first] = cv::Point(detections[i].x + detections[i].width / 2, detections[i].y + detections[i].height / 2);
                break;
            }
        }
        if (!matched) {
            unmatched_detections.push_back(i);
        }
    }

    for (int i : unmatched_detections) {
        int new_id = next_id++;
        tracked_objects[new_id] = detections[i];
        object_centers[new_id] = cv::Point(detections[i].x + detections[i].width / 2, detections[i].y + detections[i].height / 2);
    }
}

void SORT::updateObjectTracking(const std::vector<cv::Rect>& detections) {
    associateDetectionsToTrackedObjects(detections);
    tracked_ids.clear();
    for (const auto& entry : tracked_objects) {
        tracked_ids.push_back(entry.first);
    }
}

const std::vector<int>& SORT::getTrackedObjectIDs() const {
    return tracked_ids;
}


/*
void SORT::update(const std::vector<cv::Rect>& detections) {
    updateObjectTracking(detections);
}*/
std::vector<std::vector<int>> SORT::update(const std::vector<cv::Rect>& detections) {
    updateObjectTracking(detections); // Dönüþ deðerini ayarlama 
    std::vector<std::vector<int>> tracker_results;
    for (const auto& entry : tracked_objects) {
        int id = entry.first;
        const cv::Rect& box = entry.second;
        tracker_results.push_back({ box.x, box.y, box.width, box.height, id });
    }
    return tracker_results;
}