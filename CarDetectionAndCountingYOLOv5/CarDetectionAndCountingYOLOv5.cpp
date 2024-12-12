/*
#include "CarDetectionAndCountingYOLOv5.h"
#include <fstream>
#include <opencv2/opencv.hpp>

std::vector<std::string> load_class_list()
{
    std::vector<std::string> class_list;
    std::ifstream ifs("C:/Users/kaank/Desktop/Projects/Yolo5cpp/yolov5/yolov5/coco.names");
    std::string line;
    while (getline(ifs, line))
    {
        class_list.push_back(line);
    }
    return class_list;
}

void load_net(cv::dnn::Net& net, bool is_cuda)
{
    auto result = cv::dnn::readNet("C:/Users/kaank/Desktop/Projects/Yolo5cpp/yolov5/yolov5/yolov5s.onnx");
    if (is_cuda)
    {
        std::cout << "Attempting to use CUDA\n";
        result.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        result.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA_FP16);
    }
    else
    {
        std::cout << "Running on CPU\n";
        result.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        result.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    }
    net = result;
}

const std::vector<cv::Scalar> colors = { cv::Scalar(255, 255, 0), cv::Scalar(0, 255, 0), cv::Scalar(0, 255, 255), cv::Scalar(255, 0, 0) };

const float INPUT_WIDTH = 640.0;
const float INPUT_HEIGHT = 640.0;
const float SCORE_THRESHOLD = 0.2;
const float NMS_THRESHOLD = 0.4;
const float CONFIDENCE_THRESHOLD = 0.4;

struct Detection
{
    int class_id;
    float confidence;
    cv::Rect box;
};

cv::Mat format_yolov5(const cv::Mat& source)
{
    int col = source.cols;
    int row = source.rows;
    int _max = std::max(col, row);
    cv::Mat result = cv::Mat::zeros(_max, _max, CV_8UC3);
    source.copyTo(result(cv::Rect(0, 0, col, row)));
    return result;
}

void detect(cv::Mat& image, cv::dnn::Net& net, std::vector<Detection>& output, const std::vector<std::string>& className)
{
    cv::Mat blob;
    auto input_image = format_yolov5(image);
    cv::dnn::blobFromImage(input_image, blob, 1. / 255., cv::Size(INPUT_WIDTH, INPUT_HEIGHT), cv::Scalar(), true, false);
    net.setInput(blob);
    std::vector<cv::Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());

    float x_factor = input_image.cols / INPUT_WIDTH;
    float y_factor = input_image.rows / INPUT_HEIGHT;

    float* data = (float*)outputs[0].data;

    const int dimensions = 85;
    const int rows = 25200;

    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    for (int i = 0; i < rows; ++i)
    {
        float confidence = data[4];
        if (confidence >= CONFIDENCE_THRESHOLD)
        {
            float* classes_scores = data + 5;
            cv::Mat scores(1, className.size(), CV_32FC1, classes_scores);
            cv::Point class_id;
            double max_class_score;
            minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
            if (max_class_score > SCORE_THRESHOLD)
            {
                confidences.push_back(confidence);
                class_ids.push_back(class_id.x);

                float x = data[0];
                float y = data[1];
                float w = data[2];
                float h = data[3];

                int left = int((x - 0.5 * w) * x_factor);
                int top = int((y - 0.5 * h) * y_factor);
                int width = int(w * x_factor);
                int height = int(h * y_factor);
                boxes.push_back(cv::Rect(left, top, width, height));
            }
        }
        data += 85;
    }

    std::vector<int> nms_result;
    cv::dnn::NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, nms_result);
    for (int i : nms_result)
    {
        Detection result;
        result.class_id = class_ids[i];
        result.confidence = confidences[i];
        result.box = boxes[i];
        output.push_back(result);
    }
}

int main(int argc, char** argv)
{
    std::vector<std::string> class_list = load_class_list();

    std::string video_path = "C:/Users/kaank/Downloads/deneme_video.mp4"; // Replace with your video file path
    cv::VideoCapture capture(video_path);
    if (!capture.isOpened())
    {
        std::cerr << "Error opening video\n";
        return -1;
    }

    bool is_cuda = argc > 1 && strcmp(argv[1], "cuda") == 0;

    cv::dnn::Net net;
    load_net(net, is_cuda);

    int car_count = 0; // Araç sayacı

    cv::Mat frame;
    while (capture.read(frame))
    {
        if (frame.empty())
            break;

        std::vector<Detection> output;
        detect(frame, net, output, class_list);

        for (const auto& detection : output)
        {
            auto box = detection.box;
            auto classId = detection.class_id;
            const auto color = colors[classId % colors.size()];
            cv::rectangle(frame, box, color, 3);
            cv::rectangle(frame, cv::Point(box.x, box.y - 20), cv::Point(box.x + box.width, box.y), color, cv::FILLED);
            cv::putText(frame, class_list[classId].c_str(), cv::Point(box.x, box.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));

            // Eğer nesne "car" sınıfına aitse sayacı artır
            if (class_list[classId] == "car")
            {
                car_count++;
            }
        }

        cv::putText(frame, "Car Count: " + std::to_string(car_count), cv::Point(20, 50), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 255, 255), 2);
        cv::imshow("Video Output", frame);

        if (cv::waitKey(1) == 27) // Exit if ESC is pressed
            break;
    }

    std::cout << "Total cars detected: " << car_count << std::endl;

    capture.release();
    cv::destroyAllWindows();

    return 0;
}*/

/*
#include "CarDetectionAndCountingYOLOv5.h"
#include <fstream>
#include <opencv2/opencv.hpp>
#include <string>

// Fonksiyonlar ve sabitler
std::vector<std::string> load_class_list() {
    std::vector<std::string> class_list;
    std::ifstream ifs("C:/Users/kaank/Desktop/Projects/Yolo5cpp/yolov5/yolov5/coco.names"); // Yolun doğru olduğundan emin ol
    std::string line;
    while (getline(ifs, line)) {
        class_list.push_back(line);
    }
    return class_list;
}

void load_net(cv::dnn::Net& net, bool is_cuda) {
    auto result = cv::dnn::readNet("C:/Users/kaank/Desktop/Projects/Yolo5cpp/yolov5/yolov5/yolov5s.onnx");
    if (is_cuda) {
        std::cout << "Attempting to use CUDA\n";
        result.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        result.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA_FP16);
    }
    else {
        std::cout << "Running on CPU\n";
        result.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        result.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    }
    net = result;
}

cv::Mat format_yolov5(const cv::Mat& source) {
    int col = source.cols;
    int row = source.rows;
    int _max = std::max(col, row);
    cv::Mat result = cv::Mat::zeros(_max, _max, CV_8UC3);
    source.copyTo(result(cv::Rect(0, 0, col, row)));
    return result;
}

struct Detection {
    int class_id;
    float confidence;
    cv::Rect box;
};

void detect(cv::Mat& image, cv::dnn::Net& net, std::vector<Detection>& output, const std::vector<std::string>& class_names) {
    cv::Mat blob;
    auto input_image = format_yolov5(image);
    cv::dnn::blobFromImage(input_image, blob, 1. / 255., cv::Size(640, 640), cv::Scalar(), true, false);
    net.setInput(blob);
    std::vector<cv::Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());

    float x_factor = input_image.cols / 640.0;
    float y_factor = input_image.rows / 640.0;

    float* data = (float*)outputs[0].data;

    const int dimensions = 85;
    const int rows = 25200;

    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    for (int i = 0; i < rows; ++i) {
        float confidence = data[4];
        if (confidence >= 0.4) {
            float* classes_scores = data + 5;
            cv::Mat scores(1, class_names.size(), CV_32FC1, classes_scores);
            cv::Point class_id;
            double max_class_score;
            cv::minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
            if (max_class_score > 0.2) {
                confidences.push_back(confidence);
                class_ids.push_back(class_id.x);

                float x = data[0];
                float y = data[1];
                float w = data[2];
                float h = data[3];

                int left = int((x - 0.5 * w) * x_factor);
                int top = int((y - 0.5 * h) * y_factor);
                int width = int(w * x_factor);
                int height = int(h * y_factor);
                boxes.push_back(cv::Rect(left, top, width, height));
            }
        }
        data += 85;
    }

    std::vector<int> nms_result;
    cv::dnn::NMSBoxes(boxes, confidences, 0.2, 0.4, nms_result);
    for (int i : nms_result) {
        Detection result;
        result.class_id = class_ids[i];
        result.confidence = confidences[i];
        result.box = boxes[i];
        output.push_back(result);
    }
}

// Komut satırı destekli main fonksiyonu
int main(int argc, char** argv) {
    if (argc < 5) {
        std::cerr << "Usage: myprogram --input <video_path> --output <output_file>\n";
        return -1;
    }

    std::string input_path;
    std::string output_path;

    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--input" && i + 1 < argc) {
            input_path = argv[++i];
        }
        else if (std::string(argv[i]) == "--output" && i + 1 < argc) {
            output_path = argv[++i];
        }
    }

    if (input_path.empty() || output_path.empty()) {
        std::cerr << "Error: --input and --output arguments are required.\n";
        return -1;
    }

    cv::VideoCapture capture(input_path);
    if (!capture.isOpened()) {
        std::cerr << "Error opening video file: " << input_path << "\n";
        return -1;
    }

    std::ofstream output_file(output_path);
    if (!output_file.is_open()) {
        std::cerr << "Error opening output file: " << output_path << "\n";
        return -1;
    }

    std::vector<std::string> class_names = load_class_list();
    cv::dnn::Net net;
    load_net(net, false); // CPU kullanımı için

    output_file << "Frame, CarCount\n";

    cv::Mat frame;
    int frame_index = 0;

    while (capture.read(frame)) {
        if (frame.empty()) {
            break;
        }

        std::vector<Detection> detections;
        detect(frame, net, detections, class_names);

        int car_count = 0;
        for (const auto& detection : detections) {
            if (class_names[detection.class_id] == "car") {
                car_count++;
            }
        }

        output_file << frame_index << ", " << car_count << "\n";
        frame_index++;
    }

    std::cout << "Results written to: " << output_path << "\n";

    capture.release();
    output_file.close();

    return 0;
}*/

/*-------------------------------------------------*/
/*
#include "CarDetectionAndCountingYOLOv5.h"
#include <fstream>
#include <opencv2/opencv.hpp>
#include <string>
#include <chrono>  // Süre ölçümü için

// Fonksiyonlar ve sabitler
std::vector<std::string> load_class_list() {
    std::vector<std::string> class_list;
    std::ifstream ifs("C:/Users/kaank/Desktop/Projects/Yolo5cpp/yolov5/yolov5/coco.names"); // Yolun doğru olduğundan emin ol
    std::string line;
    while (getline(ifs, line)) {
        class_list.push_back(line);
    }
    return class_list;
}

void load_net(cv::dnn::Net& net, bool is_cuda) {
    auto result = cv::dnn::readNet("C:/Users/kaank/Desktop/Projects/Yolo5cpp/yolov5/yolov5/yolov5s.onnx");
    if (is_cuda) {
        std::cout << "Attempting to use CUDA\n";
        result.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        result.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA_FP16);
    }
    else {
        std::cout << "Running on CPU\n";
        result.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        result.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    }
    net = result;
}

cv::Mat format_yolov5(const cv::Mat& source) {
    int col = source.cols;
    int row = source.rows;
    int _max = std::max(col, row);
    cv::Mat result = cv::Mat::zeros(_max, _max, CV_8UC3);
    source.copyTo(result(cv::Rect(0, 0, col, row)));
    return result;
}

struct Detection {
    int class_id;
    float confidence;
    cv::Rect box;
};

void detect(cv::Mat& image, cv::dnn::Net& net, std::vector<Detection>& output, const std::vector<std::string>& class_names) {
    cv::Mat blob;
    auto input_image = format_yolov5(image);
    cv::dnn::blobFromImage(input_image, blob, 1. / 255., cv::Size(640, 640), cv::Scalar(), true, false);
    net.setInput(blob);
    std::vector<cv::Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());

    float x_factor = input_image.cols / 640.0;
    float y_factor = input_image.rows / 640.0;

    float* data = (float*)outputs[0].data;

    const int dimensions = 85;
    const int rows = 25200;

    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    for (int i = 0; i < rows; ++i) {
        float confidence = data[4];
        if (confidence >= 0.4) {
            float* classes_scores = data + 5;
            cv::Mat scores(1, class_names.size(), CV_32FC1, classes_scores);
            cv::Point class_id;
            double max_class_score;
            cv::minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
            if (max_class_score > 0.2) {
                confidences.push_back(confidence);
                class_ids.push_back(class_id.x);

                float x = data[0];
                float y = data[1];
                float w = data[2];
                float h = data[3];

                int left = int((x - 0.5 * w) * x_factor);
                int top = int((y - 0.5 * h) * y_factor);
                int width = int(w * x_factor);
                int height = int(h * y_factor);
                boxes.push_back(cv::Rect(left, top, width, height));
            }
        }
        data += 85;
    }

    std::vector<int> nms_result;
    cv::dnn::NMSBoxes(boxes, confidences, 0.2, 0.4, nms_result);
    for (int i : nms_result) {
        Detection result;
        result.class_id = class_ids[i];
        result.confidence = confidences[i];
        result.box = boxes[i];
        output.push_back(result);
    }
}

// Komut satırı destekli main fonksiyonu
int main(int argc, char** argv) {
    if (argc < 5) {
        std::cerr << "Usage: myprogram --input <video_path> --output <output_file>\n";
        return -1;
    }

    std::string input_path;
    std::string output_path;

    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--input" && i + 1 < argc) {
            input_path = argv[++i];
        }
        else if (std::string(argv[i]) == "--output" && i + 1 < argc) {
            output_path = argv[++i];
        }
    }

    if (input_path.empty() || output_path.empty()) {
        std::cerr << "Error: --input and --output arguments are required.\n";
        return -1;
    }

    cv::VideoCapture capture(input_path);
    if (!capture.isOpened()) {
        std::cerr << "Error opening video file: " << input_path << "\n";
        return -1;
    }

    std::ofstream output_file(output_path);
    if (!output_file.is_open()) {
        std::cerr << "Error opening output file: " << output_path << "\n";
        return -1;
    }

    std::vector<std::string> class_names = load_class_list();
    cv::dnn::Net net;
    load_net(net, false); // CPU kullanımı için

    output_file << "Frame, CarCount, AvgDetectionTime(ms)\n";

    cv::Mat frame;
    int frame_index = 0;
    double total_detection_time = 0; // Toplam tespit süresi
    int total_detections = 0;  // Toplam araç tespit sayısı

    while (capture.read(frame)) {
        if (frame.empty()) {
            break;
        }

        // Zaman ölçümü
        auto start_time = std::chrono::high_resolution_clock::now();

        std::vector<Detection> detections;
        detect(frame, net, detections, class_names);

        // Zaman bitişi
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> detection_duration = end_time - start_time;
        total_detection_time += detection_duration.count(); // Geçen süreyi ekle
        total_detections += detections.size();  // Toplam tespit sayısını artır

        int car_count = 0;
        for (const auto& detection : detections) {
            if (class_names[detection.class_id] == "car") {
                car_count++;
            }
        }

        // Ortalama tespit süresi hesapla
        double avg_detection_time = total_detection_time / (frame_index + 1);

        // Çıktıya yaz
        output_file << frame_index << ", " << car_count << ", " << avg_detection_time << "\n";

        frame_index++;
    }

    std::cout << "Results written to: " << output_path << "\n";

    capture.release();
    output_file.close();

    return 0;
}*/

/*-------------------------------------------------*/

/*
#include "CarDetectionAndCountingYOLOv5.h"
#include <fstream>
#include <opencv2/opencv.hpp>
#include <string>
#include <chrono>  // Süre ölçümü için

// Fonksiyonlar ve sabitler
std::vector<std::string> load_class_list() {
    std::vector<std::string> class_list;
    std::ifstream ifs("C:/Users/kaank/Desktop/Projects/Yolo5cpp/yolov5/yolov5/coco.names"); // Yolun doğru olduğundan emin ol
    std::string line;
    while (getline(ifs, line)) {
        class_list.push_back(line);
    }
    return class_list;
}

void load_net(cv::dnn::Net& net, bool is_cuda) {
    auto result = cv::dnn::readNet("C:/Users/kaank/Desktop/Projects/Yolo5cpp/yolov5/yolov5/yolov5s.onnx");
    if (is_cuda) {
        std::cout << "Attempting to use CUDA\n";
        result.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        result.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA_FP16);
    }
    else {
        std::cout << "Running on CPU\n";
        result.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        result.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    }
    net = result;
}

cv::Mat format_yolov5(const cv::Mat& source) {
    int col = source.cols;
    int row = source.rows;
    int _max = std::max(col, row);
    cv::Mat result = cv::Mat::zeros(_max, _max, CV_8UC3);
    source.copyTo(result(cv::Rect(0, 0, col, row)));
    return result;
}

struct Detection {
    int class_id;
    float confidence;
    cv::Rect box;
};

void detect(cv::Mat& image, cv::dnn::Net& net, std::vector<Detection>& output, const std::vector<std::string>& class_names) {
    cv::Mat blob;
    auto input_image = format_yolov5(image);
    cv::dnn::blobFromImage(input_image, blob, 1. / 255., cv::Size(640, 640), cv::Scalar(), true, false);
    net.setInput(blob);
    std::vector<cv::Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());

    float x_factor = input_image.cols / 640.0;
    float y_factor = input_image.rows / 640.0;

    float* data = (float*)outputs[0].data;

    const int dimensions = 85;
    const int rows = 25200;

    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    for (int i = 0; i < rows; ++i) {
        float confidence = data[4];
        if (confidence >= 0.4) {
            float* classes_scores = data + 5;
            cv::Mat scores(1, class_names.size(), CV_32FC1, classes_scores);
            cv::Point class_id;
            double max_class_score;
            cv::minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
            if (max_class_score > 0.2) {
                confidences.push_back(confidence);
                class_ids.push_back(class_id.x);

                float x = data[0];
                float y = data[1];
                float w = data[2];
                float h = data[3];

                int left = int((x - 0.5 * w) * x_factor);
                int top = int((y - 0.5 * h) * y_factor);
                int width = int(w * x_factor);
                int height = int(h * y_factor);
                boxes.push_back(cv::Rect(left, top, width, height));
            }
        }
        data += 85;
    }

    std::vector<int> nms_result;
    cv::dnn::NMSBoxes(boxes, confidences, 0.2, 0.4, nms_result);
    for (int i : nms_result) {
        Detection result;
        result.class_id = class_ids[i];
        result.confidence = confidences[i];
        result.box = boxes[i];
        output.push_back(result);
    }
}

// Komut satırı destekli main fonksiyonu
int main(int argc, char** argv) {
    if (argc < 5) {
        std::cerr << "Usage: myprogram --input <video_path> --output <output_file>\n";
        return -1;
    }

    std::string input_path;
    std::string output_path;

    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--input" && i + 1 < argc) {
            input_path = argv[++i];
        }
        else if (std::string(argv[i]) == "--output" && i + 1 < argc) {
            output_path = argv[++i];
        }
    }

    if (input_path.empty() || output_path.empty()) {
        std::cerr << "Error: --input and --output arguments are required.\n";
        return -1;
    }

    cv::VideoCapture capture(input_path);
    if (!capture.isOpened()) {
        std::cerr << "Error opening video file: " << input_path << "\n";
        return -1;
    }

    std::ofstream output_file(output_path);
    if (!output_file.is_open()) {
        std::cerr << "Error opening output file: " << output_path << "\n";
        return -1;
    }

    std::vector<std::string> class_names = load_class_list();
    cv::dnn::Net net;
    load_net(net, false); // CPU kullanımı için

    output_file << "TotalCarCount, AvgDetectionTime(ms)\n";

    cv::Mat frame;
    int frame_index = 0;
    double total_detection_time = 0; // Toplam tespit süresi
    int total_detections = 0;  // Toplam araç tespit sayısı
    int total_frames = 0;  // Toplam frame sayısı

    // Video karelerini işleme
    while (capture.read(frame)) {
        if (frame.empty()) {
            break;
        }

        // Zaman ölçümü
        auto start_time = std::chrono::high_resolution_clock::now();

        std::vector<Detection> detections;
        detect(frame, net, detections, class_names);

        // Zaman bitişi
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> detection_duration = end_time - start_time;
        total_detection_time += detection_duration.count(); // Geçen süreyi ekle
        total_detections += detections.size();  // Toplam tespit sayısını artır
        total_frames++;  // Toplam kare sayısını artır

        int car_count = 0;
        for (const auto& detection : detections) {
            if (class_names[detection.class_id] == "car") {
                car_count++;
            }
        }

        frame_index++;
    }

    // Ortalama tespit süresi hesapla
    double avg_detection_time = total_detection_time / total_frames;

    // Çıktıya toplam araç sayısı ve ortalama tespit süresi yaz
    output_file << total_detections << ", " << avg_detection_time << "\n";

    std::cout << "Results written to: " << output_path << "\n";

    capture.release();
    output_file.close();

    return 0;
}
*/

/*-------------------------------------------------*/
/*
#include "CarDetectionAndCountingYOLOv5.h"
#include <fstream>
#include <opencv2/opencv.hpp>
#include <string>
#include <chrono>  // Süre ölçümü için

// Fonksiyonlar ve sabitler
std::vector<std::string> load_class_list() {
    std::vector<std::string> class_list;
    std::ifstream ifs("C:/Users/kaank/Desktop/Projects/Yolo5cpp/yolov5/yolov5/coco.names"); // Yolun doğru olduğundan emin ol
    std::string line;
    while (getline(ifs, line)) {
        class_list.push_back(line);
    }
    return class_list;
}

void load_net(cv::dnn::Net& net, bool is_cuda) {
    auto result = cv::dnn::readNet("C:/Users/kaank/Desktop/Projects/Yolo5cpp/yolov5/yolov5/yolov5s.onnx");
    if (is_cuda) {
        std::cout << "Attempting to use CUDA\n";
        result.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        result.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA_FP16);
    }
    else {
        std::cout << "Running on CPU\n";
        result.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        result.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    }
    net = result;
}

cv::Mat format_yolov5(const cv::Mat& source) {
    int col = source.cols;
    int row = source.rows;
    int _max = std::max(col, row);
    cv::Mat result = cv::Mat::zeros(_max, _max, CV_8UC3);
    source.copyTo(result(cv::Rect(0, 0, col, row)));
    return result;
}

struct Detection {
    int class_id;
    float confidence;
    cv::Rect box;
};

void detect(cv::Mat& image, cv::dnn::Net& net, std::vector<Detection>& output, const std::vector<std::string>& class_names) {
    cv::Mat blob;
    auto input_image = format_yolov5(image);
    cv::dnn::blobFromImage(input_image, blob, 1. / 255., cv::Size(640, 640), cv::Scalar(), true, false);
    net.setInput(blob);
    std::vector<cv::Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());

    float x_factor = input_image.cols / 640.0;
    float y_factor = input_image.rows / 640.0;

    float* data = (float*)outputs[0].data;

    const int dimensions = 85;
    const int rows = 25200;

    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    for (int i = 0; i < rows; ++i) {
        float confidence = data[4];
        if (confidence >= 0.4) {
            float* classes_scores = data + 5;
            cv::Mat scores(1, class_names.size(), CV_32FC1, classes_scores);
            cv::Point class_id;
            double max_class_score;
            cv::minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
            if (max_class_score > 0.2) {
                confidences.push_back(confidence);
                class_ids.push_back(class_id.x);

                float x = data[0];
                float y = data[1];
                float w = data[2];
                float h = data[3];

                int left = int((x - 0.5 * w) * x_factor);
                int top = int((y - 0.5 * h) * y_factor);
                int width = int(w * x_factor);
                int height = int(h * y_factor);
                boxes.push_back(cv::Rect(left, top, width, height));
            }
        }
        data += 85;
    }

    std::vector<int> nms_result;
    cv::dnn::NMSBoxes(boxes, confidences, 0.2, 0.4, nms_result);
    for (int i : nms_result) {
        Detection result;
        result.class_id = class_ids[i];
        result.confidence = confidences[i];
        result.box = boxes[i];
        output.push_back(result);
    }
}

// Komut satırı destekli main fonksiyonu
int main(int argc, char** argv) {
    if (argc < 5) {
        std::cerr << "Usage: myprogram --input <video_path> --output <output_file>\n";
        return -1;
    }

    std::string input_path;
    std::string output_path;

    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--input" && i + 1 < argc) {
            input_path = argv[++i];
        }
        else if (std::string(argv[i]) == "--output" && i + 1 < argc) {
            output_path = argv[++i];
        }
    }

    if (input_path.empty() || output_path.empty()) {
        std::cerr << "Error: --input and --output arguments are required.\n";
        return -1;
    }

    cv::VideoCapture capture(input_path);
    if (!capture.isOpened()) {
        std::cerr << "Error opening video file: " << input_path << "\n";
        return -1;
    }

    std::ofstream output_file(output_path);
    if (!output_file.is_open()) {
        std::cerr << "Error opening output file: " << output_path << "\n";
        return -1;
    }

    std::vector<std::string> class_names = load_class_list();
    cv::dnn::Net net;
    load_net(net, false); // CPU kullanımı için

    output_file << "Frame, CarCount, AvgDetectionTime(ms)\n";

    cv::Mat frame;
    int frame_index = 0;
    double total_detection_time = 0; // Toplam tespit süresi
    int total_detections = 0;  // Toplam araç tespit sayısı

    while (capture.read(frame)) {
        if (frame.empty()) {
            break;
        }

        // Zaman ölçümü
        auto start_time = std::chrono::high_resolution_clock::now();

        std::vector<Detection> detections;
        detect(frame, net, detections, class_names);

        // Zaman bitişi
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> detection_duration = end_time - start_time;
        total_detection_time += detection_duration.count(); // Geçen süreyi ekle
        total_detections += detections.size();  // Toplam tespit sayısını artır

        int car_count = 0;
        for (const auto& detection : detections) {
            if (class_names[detection.class_id] == "car") {
                car_count++;
            }
        }

        // Ortalama tespit süresi hesapla
        double avg_detection_time = total_detection_time / (frame_index + 1);

        // Çıktıya yaz
        output_file << frame_index << ", " << car_count << ", " << avg_detection_time << "\n";

        frame_index++;
    }

    std::cout << "Results written to: " << output_path << "\n";

    capture.release();
    output_file.close();

    return 0;
}*/
/*-------------------------------------------------------------*/
/*
#include "CarDetectionAndCountingYOLOv5.h"
#include <fstream>
#include <opencv2/opencv.hpp>
#include <string>
#include <chrono>  // Süre ölçümü için

// Fonksiyonlar ve sabitler
std::vector<std::string> load_class_list() {
    std::vector<std::string> class_list;
    std::ifstream ifs("C:/Users/kaank/Desktop/Projects/Yolo5cpp/yolov5/yolov5/coco.names"); // Yolun doğru olduğundan emin ol
    std::string line;
    while (getline(ifs, line)) {
        class_list.push_back(line);
    }
    return class_list;
}

void load_net(cv::dnn::Net& net, bool is_cuda) {
    auto result = cv::dnn::readNet("C:/Users/kaank/Desktop/Projects/Yolo5cpp/yolov5/yolov5/yolov5s.onnx");
    if (is_cuda) {
        std::cout << "Attempting to use CUDA\n";
        result.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        result.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA_FP16);
    }
    else {
        std::cout << "Running on CPU\n";
        result.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        result.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    }
    net = result;
}

cv::Mat format_yolov5(const cv::Mat& source) {
    int col = source.cols;
    int row = source.rows;
    int _max = std::max(col, row);
    cv::Mat result = cv::Mat::zeros(_max, _max, CV_8UC3);
    source.copyTo(result(cv::Rect(0, 0, col, row)));
    return result;
}

struct Detection {
    int class_id;
    float confidence;
    cv::Rect box;
    cv::Point center;  // Merkez noktası
};

void detect(cv::Mat& image, cv::dnn::Net& net, std::vector<Detection>& output, const std::vector<std::string>& class_names) {
    cv::Mat blob;
    auto input_image = format_yolov5(image);
    cv::dnn::blobFromImage(input_image, blob, 1. / 255., cv::Size(640, 640), cv::Scalar(), true, false);
    net.setInput(blob);
    std::vector<cv::Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());

    float x_factor = input_image.cols / 640.0;
    float y_factor = input_image.rows / 640.0;

    float* data = (float*)outputs[0].data;

    const int dimensions = 85;
    const int rows = 25200;

    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    for (int i = 0; i < rows; ++i) {
        float confidence = data[4];
        if (confidence >= 0.4) {
            float* classes_scores = data + 5;
            cv::Mat scores(1, class_names.size(), CV_32FC1, classes_scores);
            cv::Point class_id;
            double max_class_score;
            cv::minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
            if (max_class_score > 0.2) {
                confidences.push_back(confidence);
                class_ids.push_back(class_id.x);

                float x = data[0];
                float y = data[1];
                float w = data[2];
                float h = data[3];

                int left = int((x - 0.5 * w) * x_factor);
                int top = int((y - 0.5 * h) * y_factor);
                int width = int(w * x_factor);
                int height = int(h * y_factor);
                boxes.push_back(cv::Rect(left, top, width, height));
            }
        }
        data += 85;
    }

    std::vector<int> nms_result;
    cv::dnn::NMSBoxes(boxes, confidences, 0.2, 0.4, nms_result);
    for (int i : nms_result) {
        Detection result;
        result.class_id = class_ids[i];
        result.confidence = confidences[i];
        result.box = boxes[i];
        result.center = cv::Point(boxes[i].x + boxes[i].width / 2, boxes[i].y + boxes[i].height / 2);  // Merkez noktası
        output.push_back(result);
    }
}

// Komut satırı destekli main fonksiyonu
int main(int argc, char** argv) {
    if (argc < 5) {
        std::cerr << "Usage: myprogram --input <video_path> --output <output_file>\n";
        return -1;
    }

    std::string input_path;
    std::string output_path;

    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--input" && i + 1 < argc) {
            input_path = argv[++i];
        }
        else if (std::string(argv[i]) == "--output" && i + 1 < argc) {
            output_path = argv[++i];
        }
    }

    if (input_path.empty() || output_path.empty()) {
        std::cerr << "Error: --input and --output arguments are required.\n";
        return -1;
    }

    cv::VideoCapture capture(input_path);
    if (!capture.isOpened()) {
        std::cerr << "Error opening video file: " << input_path << "\n";
        return -1;
    }

    std::ofstream output_file(output_path);
    if (!output_file.is_open()) {
        std::cerr << "Error opening output file: " << output_path << "\n";
        return -1;
    }

    std::vector<std::string> class_names = load_class_list();
    cv::dnn::Net net;
    load_net(net, false); // CPU kullanımı için

    output_file << "Frame, CarCount, AvgDetectionTime(ms)\n";

    cv::Mat frame;
    int frame_index = 0;
    double total_detection_time = 0; // Toplam tespit süresi
    int total_detections = 0;  // Toplam araç tespit sayısı

    // Çizgi koordinatları
    int line_y = 300; // Çizginin y koordinatı (frame'in ortası gibi)

    while (capture.read(frame)) {
        if (frame.empty()) {
            break;
        }

        // Zaman ölçümü
        auto start_time = std::chrono::high_resolution_clock::now();

        std::vector<Detection> detections;
        detect(frame, net, detections, class_names);

        // Zaman bitişi
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> detection_duration = end_time - start_time;
        total_detection_time += detection_duration.count(); // Geçen süreyi ekle
        total_detections += detections.size();  // Toplam tespit sayısını artır

        int car_count = 0;
        for (const auto& detection : detections) {
            if (class_names[detection.class_id] == "car" or class_names[detection.class_id] == "bus" or class_names[detection.class_id] == "truck") {
                car_count++;

                // Bounding box çiz
                cv::rectangle(frame, detection.box, cv::Scalar(0, 255, 0), 2);

                // Merkez noktasını işaretle
                cv::circle(frame, detection.center, 5, cv::Scalar(0, 0, 255), -1);
            }
        }

        // Ortada çizgi çiz
        cv::line(frame, cv::Point(0, line_y), cv::Point(frame.cols, line_y), cv::Scalar(255, 0, 0), 2);

        // Ortalama tespit süresi hesapla
        double avg_detection_time = total_detection_time / (frame_index + 1);

        // Çıktıya yaz
        output_file << frame_index << ", " << car_count << ", " << avg_detection_time << "\n";

        // Ekranda videoyu göster
        cv::imshow("Processed Video", frame);

        // 'Esc' tuşuna basıldığında çıkış yap
        if (cv::waitKey(1) == 27) {  // 27, 'Esc' tuşunun ASCII kodudur
            break;
        }

        frame_index++;
    }

    std::cout << "Results written to: " << output_path << "\n";

    capture.release();
    output_file.close();

    return 0;
}*/

/*-------------------------------------------------------------*/
/*
#include "CarDetectionAndCountingYOLOv5.h"
#include <fstream>
#include <opencv2/opencv.hpp>
#include <string>
#include <chrono>  // Süre ölçümü için

// Fonksiyonlar ve sabitler
std::vector<std::string> load_class_list() {
    std::vector<std::string> class_list;
    std::ifstream ifs("C:/Users/kaank/Desktop/Projects/Yolo5cpp/yolov5/yolov5/coco.names"); // Yolun doğru olduğundan emin ol
    std::string line;
    while (getline(ifs, line)) {
        class_list.push_back(line);
    }
    return class_list;
}

void load_net(cv::dnn::Net& net, bool is_cuda) {
    auto result = cv::dnn::readNet("C:/Users/kaank/Desktop/Projects/Yolo5cpp/yolov5/yolov5/yolov5s.onnx");
    if (is_cuda) {
        std::cout << "Attempting to use CUDA\n";
        result.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        result.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA_FP16);
    }
    else {
        std::cout << "Running on CPU\n";
        result.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        result.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    }
    net = result;
}

cv::Mat format_yolov5(const cv::Mat& source) {
    int col = source.cols;
    int row = source.rows;
    int _max = std::max(col, row);
    cv::Mat result = cv::Mat::zeros(_max, _max, CV_8UC3);
    source.copyTo(result(cv::Rect(0, 0, col, row)));
    return result;
}

struct Detection {
    int class_id;
    float confidence;
    cv::Rect box;
    cv::Point center;  // Merkez noktası
    bool passed_line;  // Çizgiyi geçip geçmediğini takip et
};

void detect(cv::Mat& image, cv::dnn::Net& net, std::vector<Detection>& output, const std::vector<std::string>& class_names) {
    cv::Mat blob;
    auto input_image = format_yolov5(image);
    cv::dnn::blobFromImage(input_image, blob, 1. / 255., cv::Size(640, 640), cv::Scalar(), true, false);
    net.setInput(blob);
    std::vector<cv::Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());

    float x_factor = input_image.cols / 640.0;
    float y_factor = input_image.rows / 640.0;

    float* data = (float*)outputs[0].data;

    const int dimensions = 85;
    const int rows = 25200;

    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    for (int i = 0; i < rows; ++i) {
        float confidence = data[4];
        if (confidence >= 0.4) {
            float* classes_scores = data + 5;
            cv::Mat scores(1, class_names.size(), CV_32FC1, classes_scores);
            cv::Point class_id;
            double max_class_score;
            cv::minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
            if (max_class_score > 0.2) {
                confidences.push_back(confidence);
                class_ids.push_back(class_id.x);

                float x = data[0];
                float y = data[1];
                float w = data[2];
                float h = data[3];

                int left = int((x - 0.5 * w) * x_factor);
                int top = int((y - 0.5 * h) * y_factor);
                int width = int(w * x_factor);
                int height = int(h * y_factor);
                boxes.push_back(cv::Rect(left, top, width, height));
            }
        }
        data += 85;
    }

    std::vector<int> nms_result;
    cv::dnn::NMSBoxes(boxes, confidences, 0.2, 0.4, nms_result);
    for (int i : nms_result) {
        Detection result;
        result.class_id = class_ids[i];
        result.confidence = confidences[i];
        result.box = boxes[i];
        result.center = cv::Point(boxes[i].x + boxes[i].width / 2, boxes[i].y + boxes[i].height / 2);  // Merkez noktası
        result.passed_line = false;  // Başlangıçta geçiş yapılmadı
        output.push_back(result);
    }
}

// Komut satırı destekli main fonksiyonu
int main(int argc, char** argv) {
    if (argc < 5) {
        std::cerr << "Usage: myprogram --input <video_path> --output <output_file>\n";
        return -1;
    }

    std::string input_path;
    std::string output_path;

    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--input" && i + 1 < argc) {
            input_path = argv[++i];
        }
        else if (std::string(argv[i]) == "--output" && i + 1 < argc) {
            output_path = argv[++i];
        }
    }

    if (input_path.empty() || output_path.empty()) {
        std::cerr << "Error: --input and --output arguments are required.\n";
        return -1;
    }

    cv::VideoCapture capture(input_path);
    if (!capture.isOpened()) {
        std::cerr << "Error opening video file: " << input_path << "\n";
        return -1;
    }

    std::ofstream output_file(output_path);
    if (!output_file.is_open()) {
        std::cerr << "Error opening output file: " << output_path << "\n";
        return -1;
    }

    std::vector<std::string> class_names = load_class_list();
    cv::dnn::Net net;
    load_net(net, false); // CPU kullanımı için

    output_file << "Frame, CarCount, AvgDetectionTime(ms)\n";

    cv::Mat frame;
    int frame_index = 0;
    double total_detection_time = 0; // Toplam tespit süresi
    int total_detections = 0;  // Toplam araç tespit sayısı

    // Çizgi koordinatları
    int line_y = 300; // Çizginin y koordinatı (frame'in ortası gibi)

    while (capture.read(frame)) {
        if (frame.empty()) {
            break;
        }

        // Zaman ölçümü
        auto start_time = std::chrono::high_resolution_clock::now();

        std::vector<Detection> detections;
        detect(frame, net, detections, class_names);

        // Zaman bitişi
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> detection_duration = end_time - start_time;
        total_detection_time += detection_duration.count(); // Geçen süreyi ekle
        total_detections += detections.size();  // Toplam tespit sayısını artır

        int car_count = 0;
        for (auto& detection : detections) {
            if (class_names[detection.class_id] == "car") {
                // Bounding box çiz
                cv::rectangle(frame, detection.box, cv::Scalar(0, 255, 0), 2);

                // Merkez noktasını işaretle
                cv::circle(frame, detection.center, 5, cv::Scalar(0, 0, 255), -1);

                // Çizgiyi geçme kontrolü
                if (!detection.passed_line && detection.center.y > line_y) {
                    detection.passed_line = true;  // Araba çizgiyi geçti
                    car_count++;  // Sayımı artır
                }
            }
        }

        // Ortada çizgi çiz
        cv::line(frame, cv::Point(0, line_y), cv::Point(frame.cols, line_y), cv::Scalar(255, 0, 0), 2);

        // Ortalama tespit süresi hesapla
        double avg_detection_time = total_detection_time / (frame_index + 1);

        // Çıktıya yaz
        output_file << frame_index << ", " << car_count << ", " << avg_detection_time << "\n";

        // Ekranda videoyu göster
        cv::imshow("Processed Video", frame);

        // 'Esc' tuşuna basıldığında çıkış yap
        if (cv::waitKey(1) == 27) {  // 27, 'Esc' tuşunun ASCII kodudur
            break;
        }

        frame_index++;
    }

    std::cout << "Results written to: " << output_path << "\n";

    capture.release();
    output_file.close();

    return 0;
}

*/

/*-------------------------------------------------------------*/
/*
#include "CarDetectionAndCountingYOLOv5.h"
#include <fstream>
#include <opencv2/opencv.hpp>
#include <string>
#include <chrono>  // Süre ölçümü için
#include <unordered_map>  // ID yönetimi için
#include <vector>  // cars_crossed için

// Fonksiyonlar ve sabitler
std::vector<std::string> load_class_list() {
    std::vector<std::string> class_list;
    std::ifstream ifs("C:/Users/kaank/Desktop/Projects/Yolo5cpp/yolov5/yolov5/coco.names");
    std::string line;
    while (getline(ifs, line)) {
        class_list.push_back(line);
    }
    return class_list;
}

void load_net(cv::dnn::Net& net, bool is_cuda) {
    auto result = cv::dnn::readNet("C:/Users/kaank/Desktop/Projects/Yolo5cpp/yolov5/yolov5/yolov5s.onnx");
    if (is_cuda) {
        std::cout << "Attempting to use CUDA\n";
        result.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        result.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA_FP16);
    }
    else {
        std::cout << "Running on CPU\n";
        result.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        result.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    }
    net = result;
}

cv::Mat format_yolov5(const cv::Mat& source) {
    int col = source.cols;
    int row = source.rows;
    int _max = std::max(col, row);
    cv::Mat result = cv::Mat::zeros(_max, _max, CV_8UC3);
    source.copyTo(result(cv::Rect(0, 0, col, row)));
    return result;
}

struct Detection {
    int class_id;
    float confidence;
    cv::Rect box;
    int id;  // Her araba için bir ID
    bool crossed;  // Araba çizgiyi geçip geçmedi
};

void detect(cv::Mat& image, cv::dnn::Net& net, std::vector<Detection>& output, const std::vector<std::string>& class_names) {
    cv::Mat blob;
    auto input_image = format_yolov5(image);
    cv::dnn::blobFromImage(input_image, blob, 1. / 255., cv::Size(640, 640), cv::Scalar(), true, false);
    net.setInput(blob);
    std::vector<cv::Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());

    float x_factor = input_image.cols / 640.0;
    float y_factor = input_image.rows / 640.0;

    float* data = (float*)outputs[0].data;

    const int dimensions = 85;
    const int rows = 25200;

    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    for (int i = 0; i < rows; ++i) {
        float confidence = data[4];
        if (confidence >= 0.4) {
            float* classes_scores = data + 5;
            cv::Mat scores(1, class_names.size(), CV_32FC1, classes_scores);
            cv::Point class_id;
            double max_class_score;
            cv::minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
            if (max_class_score > 0.2) {
                confidences.push_back(confidence);
                class_ids.push_back(class_id.x);

                float x = data[0];
                float y = data[1];
                float w = data[2];
                float h = data[3];

                int left = int((x - 0.5 * w) * x_factor);
                int top = int((y - 0.5 * h) * y_factor);
                int width = int(w * x_factor);
                int height = int(h * y_factor);
                boxes.push_back(cv::Rect(left, top, width, height));
            }
        }
        data += 85;
    }

    std::vector<int> nms_result;
    cv::dnn::NMSBoxes(boxes, confidences, 0.8, 0.8, nms_result);
    for (int i : nms_result) {
        Detection result;
        result.class_id = class_ids[i];
        result.confidence = confidences[i];
        result.box = boxes[i];
        result.crossed = false;  // Başlangıçta çizgiyi geçmedi
        output.push_back(result);
    }
}

// Komut satırı destekli main fonksiyonu
int main(int argc, char** argv) {
    if (argc < 5) {
        std::cerr << "Usage: myprogram --input <video_path> --output <output_file>\n";
        return -1;
    }

    std::string input_path;
    std::string output_path;

    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--input" && i + 1 < argc) {
            input_path = argv[++i];
        }
        else if (std::string(argv[i]) == "--output" && i + 1 < argc) {
            output_path = argv[++i];
        }
    }

    if (input_path.empty() || output_path.empty()) {
        std::cerr << "Error: --input and --output arguments are required.\n";
        return -1;
    }

    cv::VideoCapture capture(input_path);
    if (!capture.isOpened()) {
        std::cerr << "Error opening video file: " << input_path << "\n";
        return -1;
    }

    std::ofstream output_file(output_path);
    if (!output_file.is_open()) {
        std::cerr << "Error opening output file: " << output_path << "\n";
        return -1;
    }

    std::vector<std::string> class_names = load_class_list();
    cv::dnn::Net net;
    load_net(net, false); // CPU kullanımı için

    output_file << "Frame, CarCount, AvgDetectionTime(ms)\n";

    cv::Mat frame;
    int frame_index = 0;
    double total_detection_time = 0; // Toplam tespit süresi
    int total_detections = 0;  // Toplam araç tespit sayısı
    int total_passed_cars = 0;  // Geçen toplam araç sayısı
    std::unordered_map<int, cv::Rect> car_positions; // ID ve konumları tutmak için
    std::unordered_map<int, bool> car_crossed; // Geçiş kontrolü için
    int next_id = 0; // Yeni ID'yi tutmak için
    std::vector<int> cars_crossed; // Çizgiyi geçip geçmeyen arabaları tutacak vektör

    // Video işleme döngüsü
    while (capture.read(frame)) {
        if (frame.empty()) {
            break;
        }

        // Çizgi koordinatları: Çizgi, frame'in tam ortasında olacak
        int line_y = frame.rows / 2;
        cv::line(frame, cv::Point(0, line_y), cv::Point(frame.cols, line_y), cv::Scalar(255, 0, 0), 5); // Çizgi kalınlığını 5 yaptık

        // Zaman ölçümü
        auto start_time = std::chrono::high_resolution_clock::now();

        std::vector<Detection> detections;
        detect(frame, net, detections, class_names);

        // Zaman bitişi
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> detection_duration = end_time - start_time;
        total_detection_time += detection_duration.count(); // Geçen süreyi ekle
        total_detections += detections.size();  // Toplam tespit sayısını artır

        int car_count = 0;
        for (auto& detection : detections) {
            if (class_names[detection.class_id] == "car") {
                // Daha önce bu arabaya bir ID atandı mı?
                bool found = false;
                for (auto& [id, rect] : car_positions) {
                    // Eğer ID'li bir araç, mevcut tespitle aynı konumda ise, ID'yi kullan
                    if ((rect & detection.box).area() > 0) { // Konumlar çakışıyorsa
                        detection.id = id;
                        found = true;
                        break;
                    }
                }

                // Eğer bulunmadıysa yeni bir ID ata
                if (!found) {
                    detection.id = next_id++;
                    car_positions[detection.id] = detection.box;
                    car_crossed[detection.id] = false; // Yeni araba geçiş yapmadı
                }

                // Araba tespiti yapılınca bounding box'ı çizelim
                cv::rectangle(frame, detection.box, cv::Scalar(0, 255, 0), 10);  // Bounding box kalınlığını 6 yaptık

                // Araba ID'sini yaz
                cv::putText(frame, std::to_string(detection.id), cv::Point(detection.box.x, detection.box.y - 10),
                    cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 255), 10);  // ID'nin görünürlüğünü artırdık

                cv::Point center(detection.box.x + detection.box.width / 2, detection.box.y + detection.box.height / 2);
                cv::circle(frame, center, 8, cv::Scalar(0, 0, 255), -1); // Araba merkezini işaretle, biraz daha büyük yaptık

                // Çizgiyi geçen arabaları sayalım
                if (center.y > line_y && !car_crossed[detection.id]) {
                    total_passed_cars++;
                    car_crossed[detection.id] = true; // Bu araba çizgiyi geçti
                    cars_crossed.push_back(detection.id); // Bu arabayı geçmiş olarak kaydediyoruz
                }

                car_count++;
            }
        }

        // Ortalama tespit süresi hesapla
        double avg_detection_time = total_detection_time / (frame_index + 1);

        // Çıktıya yaz
        output_file << frame_index << ", " << total_passed_cars << ", " << avg_detection_time << "\n";

        // Videoyu 640x640 boyutunda ekranda göster
        cv::Mat resized_frame;
        cv::resize(frame, resized_frame, cv::Size(640, 640));  // Boyutlandırma
        cv::imshow("Car Detection", resized_frame); // Videoyu ekranda göster

        // Ekranda geçen araç sayısını ve tespit süresini göster
        std::string text = "Cars passed: " + std::to_string(total_passed_cars) + " Avg detection time: " + std::to_string(avg_detection_time) + " ms";
        cv::putText(resized_frame, text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);

        // 'q' tuşuna basıldığında çıkış
        if (cv::waitKey(1) == 'q') {
            break;
        }

        frame_index++;
    }

    std::cout << "Results written to: " << output_path << "\n";

    capture.release();
    output_file.close();
    cv::destroyAllWindows(); // Pencereyi kapat

    return 0;
}
*/


/*---------------------------------------------------------------------------*/
#include "CarDetectionAndCountingYOLOv5.h"
#include <fstream>
#include <opencv2/opencv.hpp>
#include <string>
#include <chrono>  // Süre ölçümü için
#include <unordered_map>  // ID yönetimi için
#include <vector>  // cars_crossed için

// Fonksiyonlar ve sabitler
std::vector<std::string> load_class_list() {
    std::vector<std::string> class_list;
    std::ifstream ifs("C:/Users/kaank/Desktop/Projects/Yolo5cpp/yolov5/yolov5/coco.names");
    std::string line;
    while (getline(ifs, line)) {
        class_list.push_back(line);
    }
    return class_list;
}

void load_net(cv::dnn::Net& net, bool is_cuda) {
    auto result = cv::dnn::readNet("C:/Users/kaank/Desktop/Projects/Yolo5cpp/yolov5/yolov5/yolov5s.onnx");
    if (is_cuda) {
        std::cout << "Attempting to use CUDA\n";
        result.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        result.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA_FP16);
    }
    else {
        std::cout << "Running on CPU\n";
        result.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        result.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    }
    net = result;
}

cv::Mat format_yolov5(const cv::Mat& source) {
    int col = source.cols;
    int row = source.rows;
    int _max = std::max(col, row);
    cv::Mat result = cv::Mat::zeros(_max, _max, CV_8UC3);
    source.copyTo(result(cv::Rect(0, 0, col, row)));
    return result;
}

struct Detection {
    int class_id;
    float confidence;
    cv::Rect box;
    int id;  // Her araba için bir ID
    bool crossed;  // Araba çizgiyi geçip geçmedi
};

void detect(cv::Mat& image, cv::dnn::Net& net, std::vector<Detection>& output, const std::vector<std::string>& class_names) {
    cv::Mat blob;
    auto input_image = format_yolov5(image);
    cv::dnn::blobFromImage(input_image, blob, 1. / 255., cv::Size(640, 640), cv::Scalar(), true, false);
    net.setInput(blob);
    std::vector<cv::Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());

    float x_factor = input_image.cols / 640.0;
    float y_factor = input_image.rows / 640.0;

    float* data = (float*)outputs[0].data;

    const int dimensions = 85;
    const int rows = 25200;

    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    for (int i = 0; i < rows; ++i) {
        float confidence = data[4];
        if (confidence >= 0.4) {
            float* classes_scores = data + 5;
            cv::Mat scores(1, class_names.size(), CV_32FC1, classes_scores);
            cv::Point class_id;
            double max_class_score;
            cv::minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
            if (max_class_score > 0.2) {
                confidences.push_back(confidence);
                class_ids.push_back(class_id.x);

                float x = data[0];
                float y = data[1];
                float w = data[2];
                float h = data[3];

                int left = int((x - 0.5 * w) * x_factor);
                int top = int((y - 0.5 * h) * y_factor);
                int width = int(w * x_factor);
                int height = int(h * y_factor);
                boxes.push_back(cv::Rect(left, top, width, height));
            }
        }
        data += 85;
    }

    std::vector<int> nms_result;
    cv::dnn::NMSBoxes(boxes, confidences, 0.8, 0.8, nms_result);
    for (int i : nms_result) {
        Detection result;
        result.class_id = class_ids[i];
        result.confidence = confidences[i];
        result.box = boxes[i];
        result.crossed = false;  // Başlangıçta çizgiyi geçmedi
        output.push_back(result);
    }
}

// Komut satırı destekli main fonksiyonu
int main(int argc, char** argv) {
    if (argc < 5) {
        std::cerr << "Usage: myprogram --input <video_path> --output <output_file>\n";
        return -1;
    }

    std::string input_path;
    std::string output_path;

    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--input" && i + 1 < argc) {
            input_path = argv[++i];
        }
        else if (std::string(argv[i]) == "--output" && i + 1 < argc) {
            output_path = argv[++i];
        }
    }

    if (input_path.empty() || output_path.empty()) {
        std::cerr << "Error: --input and --output arguments are required.\n";
        return -1;
    }

    cv::VideoCapture capture(input_path);
    if (!capture.isOpened()) {
        std::cerr << "Error opening video file: " << input_path << "\n";
        return -1;
    }

    std::ofstream output_file(output_path);
    if (!output_file.is_open()) {
        std::cerr << "Error opening output file: " << output_path << "\n";
        return -1;
    }

    std::vector<std::string> class_names = load_class_list();
    cv::dnn::Net net;
    load_net(net, false); // CPU kullanımı için

    output_file << "Frame, CarCount, AvgDetectionTime(ms)\n";

    cv::Mat frame;
    int frame_index = 0;
    double total_detection_time = 0; // Toplam tespit süresi
    int total_detections = 0;  // Toplam araç tespit sayısı
    int total_passed_cars = 0;  // Geçen toplam araç sayısı
    std::unordered_map<int, cv::Rect> car_positions; // ID ve konumları tutmak için
    std::unordered_map<int, bool> car_crossed; // Geçiş kontrolü için
    int next_id = 0; // Yeni ID'yi tutmak için
    std::vector<int> cars_crossed; // Çizgiyi geçip geçmeyen arabaları tutacak vektör

    // Video işleme döngüsü
    while (capture.read(frame)) {
        if (frame.empty()) {
            break;
        }

        // Çizgi koordinatları: Çizgi, frame'in tam ortasında olacak
        int line_y = frame.rows / 2;
        cv::line(frame, cv::Point(0, line_y), cv::Point(frame.cols, line_y), cv::Scalar(255, 0, 0), 5); // Çizgi kalınlığını 5 yaptık

        // Zaman ölçümü
        auto start_time = std::chrono::high_resolution_clock::now();

        std::vector<Detection> detections;
        detect(frame, net, detections, class_names);

        // Zaman bitişi
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> detection_duration = end_time - start_time;
        total_detection_time += detection_duration.count(); // Geçen süreyi ekle
        total_detections += detections.size();  // Toplam tespit sayısını artır

        int car_count = 0;
        for (auto& detection : detections) {
            if (class_names[detection.class_id] == "car") {
                // Daha önce bu arabaya bir ID atandı mı?
                bool found = false;
                for (auto& [id, rect] : car_positions) {
                    // Eğer ID'li bir araç, mevcut tespitle aynı konumda ise, ID'yi kullan
                    if ((rect & detection.box).area() > 0) { // Konumlar çakışıyorsa
                        detection.id = id;
                        found = true;
                        break;
                    }
                }

                // Eğer bulunmadıysa yeni bir ID ata
                if (!found) {
                    detection.id = next_id++;
                    car_positions[detection.id] = detection.box;
                    car_crossed[detection.id] = false; // Yeni araba geçiş yapmadı
                }

                // Araba tespiti yapılınca bounding box'ı çizelim
                cv::rectangle(frame, detection.box, cv::Scalar(0, 255, 0), 10);  // Bounding box kalınlığını 6 yaptık

                // Araba ID'sini yaz
                cv::putText(frame, std::to_string(detection.id), cv::Point(detection.box.x, detection.box.y - 10),
                    cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 255), 10);  // ID'nin görünürlüğünü artırdık

                cv::Point center(detection.box.x + detection.box.width / 2, detection.box.y + detection.box.height / 2);
                cv::circle(frame, center, 8, cv::Scalar(0, 0, 255), -1); // Araba merkezini işaretle, biraz daha büyük yaptık

                // Çizgiyi geçen arabaları sayalım
                if (center.y > line_y && !car_crossed[detection.id]) {
                    total_passed_cars++;
                    car_crossed[detection.id] = true; // Bu araba çizgiyi geçti
                    cars_crossed.push_back(detection.id); // Bu arabayı geçmiş olarak kaydediyoruz
                }

                car_count++;
            }
        }

        // Ortalama tespit süresi hesapla
        double avg_detection_time = total_detection_time / (frame_index + 1);

        // Çıktıya yaz
        output_file << frame_index << ", " << total_passed_cars << ", " << avg_detection_time << "\n";

        // Videoyu 640x640 boyutunda ekranda göster
        cv::Mat resized_frame;
        // Sol üst köşede araç sayısını yaz
        cv::resize(frame, resized_frame, cv::Size(640, 640));
        std::string car_count_text = "Cars Passed: " + std::to_string(total_passed_cars);
        cv::putText(resized_frame, car_count_text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 255, 255), 5);// Boyutlandırma
        // Sağ üst köşede ortalama tespit süresi yaz
        std::string avg_time_text = "Avg Time: " + std::to_string(avg_detection_time) + " ms";
        cv::putText(resized_frame, avg_time_text, cv::Point(resized_frame.cols - 300, 30), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 255, 255), 5);
        cv::imshow("Car Detection", resized_frame); // Videoyu ekranda göster

       

       
        // 'q' tuşuna basıldığında çıkış
        if (cv::waitKey(1) == 'q') {
            break;
        }

        frame_index++;
    }

    capture.release();
    output_file.close();

    return 0;
}









