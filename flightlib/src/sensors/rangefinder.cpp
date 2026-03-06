#include "flightlib/sensors/rangefinder.hpp"

namespace flightlib {

Rangefinder::Rangefinder() {}

Rangefinder::~Rangefinder() {}

bool Rangefinder::feedImageQueue(const int image_layer,
                               const cv::Mat& image_mat) {
  queue_mutex_.lock();
  
  if (depth_queue_.size() > queue_size_) depth_queue_.resize(queue_size_);
  depth_queue_.push_back(image_mat);
      
  queue_mutex_.unlock();
  return true;
}

bool Rangefinder::setRelPose(const Ref<Vector<3>> B_r_BL,
                           const Ref<Matrix<3, 3>> R_BL) {
  if (!B_r_BL.allFinite() || !R_BL.allFinite()) {
    logger_.error(
      "The setting value for Camera Relative Pose Matrix is not valid, discard "
      "the setting.");
    return false;
  }
  B_r_BL_ = B_r_BL;
  T_BL_.block<3, 3>(0, 0) = R_BL;
  T_BL_.block<3, 1>(0, 3) = B_r_BL;
  T_BL_.row(3) << 0.0, 0.0, 0.0, 1.0;
  return true;
}

void Rangefinder::setNumBeams(const int num_beams) { num_beams_ = num_beams; }
void Rangefinder::setMaxDistance(const int max_dist) { max_distance_ = max_dist; }
void Rangefinder::setStartAngle(const int start_ang) { start_angle_ = start_ang; }
void Rangefinder::setEndAngle(const int end_ang) { end_angle_ = end_ang; }

Matrix<4, 4> Rangefinder::getRelPose(void) const { 
  std::cout << "lidar get rel pose worked ----- " << std::endl;
  return T_BL_; }

bool Rangefinder::getDepthMap(cv::Mat& depth_map) {
  if (!depth_queue_.empty()) {
    depth_map = depth_queue_.front();
    depth_queue_.pop_front();
    return true;
  }
  return false;
}

int Rangefinder::getNumBeams(void) const { return num_beams_; }
int Rangefinder::getMaxDistance(void) const { return max_distance_; }
int Rangefinder::getStartAngle(void) { return start_angle_; }
int Rangefinder::getEndAngle(void) { return end_angle_; }

bool Rangefinder::getLatestDepthMap(cv::Mat& depth_map) {
  if (!depth_queue_.empty()) {
    depth_map = depth_queue_.back();
    depth_queue_.pop_back();
    return true;
  }
  return false;
}

}  // namespace flightlib
