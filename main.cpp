#include <algorithm>
#include <array>
#include <cstdio>
#include <iostream>
#include <iterator>
#include <vector>

#include <opencv2/core/mat.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/videoio.hpp>

struct ColorGray {
  cv::Mat color;
  cv::Mat gray;
};

std::tuple<int, double> next_frame(const std::vector<ColorGray> &cg_images,
                                   const cv::Mat &old_gray);

int main() {

  cv::VideoCapture cap = cv::VideoCapture("../corrupted_video.mp4");

  if (!cap.isOpened()) {
    printf("Cannot open the video file.\n");
    return -1;
  }

  int width = cap.get(cv::VideoCaptureProperties::CAP_PROP_FRAME_WIDTH);
  int height = cap.get(cv::VideoCaptureProperties::CAP_PROP_FRAME_HEIGHT);
  int fps = cap.get(cv::VideoCaptureProperties::CAP_PROP_FPS);
  int frame_count = cap.get(cv::VideoCaptureProperties::CAP_PROP_FRAME_COUNT);

  printf("size: (%d, %d), fps: %d, frame_count: %d\n", width, height, fps,
         frame_count);

  std::vector<cv::Mat> images;

  while (cap.isOpened()) {
    cv::Mat frame;

    if (!cap.read(frame)) {
      break;
    }
    images.push_back(frame);
  }
  cap.release();

  int hist_size[] = {8, 8, 8};
  int channels[] = {0, 1, 2};
  float range[] = {0, 256};
  const float *ranges[] = {range, range, range};

  std::vector<cv::MatND> hists;
  for (const auto &image : images) {
    cv::Mat color_image;
    cv::cvtColor(image, color_image, cv::COLOR_BGR2YUV);

    cv::MatND hist;
    cv::calcHist(&color_image, 1, channels, cv::Mat(), hist, 2, hist_size,
                 ranges);
    hists.push_back(hist);
  }

  int previous_size = images.size();
  int index = 0;

  auto image_is_alone = [&hists, &index](const cv::Mat &image) -> bool {
    auto get_good_hists = [&hists, index](const cv::MatND &hist) -> bool {
      return cv::compareHist(hist, hists[index], cv::HISTCMP_CORREL) > 0.8;
    };

    int count = std::count_if(hists.begin(), hists.end(), get_good_hists);
    index++;
    return ((double)count / hists.size() < 0.5);
  };

  images.erase(std::remove_if(images.begin(), images.end(), image_is_alone),
               images.end());

  std::cout << "removed " << (previous_size - images.size())
            << " images with histogram comparisons\n";

  constexpr int DIVIDOR = 20;

  printf("resized (%d, %d) to (%d, %d)\n", width, height, width / DIVIDOR,
         height / DIVIDOR);

  auto color_to_color_gray = [=](const cv::Mat &image) {
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    cv::resize(gray, gray, cv::Size(width / DIVIDOR, height / DIVIDOR));
    return ColorGray{image, gray};
  };

  std::vector<ColorGray> cg_images;
  std::transform(images.cbegin(), images.cend(), std::back_inserter(cg_images),
                 color_to_color_gray);

  std::vector<ColorGray> ordored;

  ordored.insert(ordored.begin(), cg_images[0]);
  cg_images.erase(cg_images.begin());

  int top_index{}, bot_index{};
  double top_dist{}, bot_dist{};

  std::tie(top_index, top_dist) = next_frame(cg_images, ordored.front().gray);

  ordored.insert(ordored.begin(), cg_images[top_index]);
  cg_images.erase(cg_images.begin() + top_index);

  std::tie(top_index, top_dist) = next_frame(cg_images, ordored.front().gray);
  std::tie(bot_index, bot_dist) = next_frame(cg_images, ordored.back().gray);

  while (cg_images.size()) {
    if (top_dist < bot_dist) {
      ordored.insert(ordored.begin(), cg_images[top_index]);
      cg_images.erase(cg_images.begin() + top_index);

      if (cg_images.size()) {
        if (bot_index >= top_index && bot_index != 0) {
          bot_index -= 1;
        }
        std::tie(top_index, top_dist) =
            next_frame(cg_images, ordored.front().gray);
      }
    } else {
      ordored.push_back(cg_images[bot_index]);
      cg_images.erase(cg_images.begin() + bot_index);

      if (cg_images.size()) {
        if (top_index >= bot_index && top_index != 0) {
          top_index -= 1;
        }
        std::tie(bot_index, bot_dist) =
            next_frame(cg_images, ordored.back().gray);
      }
    }
    const int percentage = 100 * (float)ordored.size() / images.size();
    std::cout << "\rcompute optical flow: " << percentage << "%" << std::flush;
  }

  std::cout << '\n';

  /*
  for (const auto &i : ordored) {
    cv::imshow("dzdqd", i.color);
    cv::waitKey(0);
  }
  */

  auto filename = "../cpp_video.mp4";
  auto fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
  auto size = cv::Size(width, height);
  auto out = cv::VideoWriter(filename, fourcc, fps, size);
  std::cout << "writing file " << filename << " ..." << '\n';
  for (const auto &cg_image : ordored) {
    out.write(cg_image.color);
  }
  out.release();
  cv::destroyAllWindows();

  return 0;
}

std::tuple<int, double> next_frame(const std::vector<ColorGray> &cg_images,
                                   const cv::Mat &old_gray) {
  static int maxCorners = 50;
  static double qualityLevel = 0.05;
  static double minDistance = 2;
  static int blockSize = 7;

  std::vector<cv::Point2f> old_corners;
  cv::goodFeaturesToTrack(old_gray, old_corners, maxCorners, qualityLevel,
                          minDistance, cv::noArray(), blockSize);

  double min_dist = 1 << 20;
  int min_index = -1;

  for (int index = 0; index < cg_images.size(); ++index) {
    const auto &new_gray = cg_images[index].gray;

    static auto winSize = cv::Size(15, 15);
    static int maxLevel = 2;
    static auto criteria = cv::TermCriteria(
        cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 10, 0.03);

    std::vector<uchar> status;
    std::vector<float> err;
    std::vector<cv::Point2f> new_corners;
    cv::calcOpticalFlowPyrLK(old_gray, new_gray, old_corners, new_corners,
                             status, err, winSize, maxLevel, criteria);

    double distance = 0;
    for (int j = 0; j < status.size(); ++j) {
      if (status[j] != 1)
        continue;

      const auto &old_corner = old_corners[j];
      const auto &new_corner = new_corners[j];

      distance += sqrt(pow(old_corner.x - new_corner.x, 2) +
                       pow(old_corner.y - new_corner.y, 2));
    }
    distance /= status.size();

    if (distance < min_dist) {
      min_dist = distance;
      min_index = index;
    }
  }
  return std::make_tuple(min_index, min_dist);
}
