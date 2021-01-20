#include <iostream>
#include <opencv2/core/mat.hpp>

int main() {
  cv::Mat hist;
  std::cout << hist.size().width << ", " << hist.size().height << "\n";
  return 0;
}