#pragma once

#include <yaml-cpp/yaml.h>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "flightlib/common/logger.hpp"
#include "flightlib/common/types.hpp"
#include "flightlib/sensors/sensor_base.hpp"

namespace flightlib {

class Rangefinder : SensorBase {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Rangefinder();
  ~Rangefinder();

  // public set functions
  bool setRelPose(const Ref<Vector<3>> B_r_BL, const Ref<Matrix<3, 3>> R_BL);
  bool feedImageQueue(const int image_layer, const cv::Mat& image_mat);
  void setNumBeams(const int num_beams);
  void setMaxDistance(const int max_dist);
  void setStartAngle(const int start_ang);
  void setEndAngle(const int end_ang);

  // public get functions
  Matrix<4, 4> getRelPose(void) const;
  int getNumBeams(void) const;
  int getMaxDistance(void) const;
  int getStartAngle(void);
  int getEndAngle(void);
  bool getDepthMap(cv::Mat& depth_map);
  bool getLatestDepthMap(cv::Mat& depth_map);

 private:
  Logger logger_{"Rangefinder"};

  // Rangefinder parameters
  int num_beams_;
  Scalar max_distance_;
  Scalar start_angle_;
  Scalar end_angle_;

  // Rangefinder relative
  Vector<3> B_r_BL_;
  Matrix<4, 4> T_BL_;

  // image data buffer
  std::mutex queue_mutex_;
  const int queue_size_ = 1;

  std::deque<cv::Mat> depth_queue_;

};

}  // namespace flightlib
