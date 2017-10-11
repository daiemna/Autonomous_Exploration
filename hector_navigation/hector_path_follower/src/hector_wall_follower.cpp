#include <hector_wall_follower/hector_wall_follower.h>

#define SIGNIFACANT_LASER_COUNT 3
// these indeices corrospond to 'scan_unified' only
// #define LEFT_LASERS_INDICIES  {495, 497, 499, 502, 504, 576, 578, 580, 582}
// #define RIGHT_LASERS_INDICIES {138, 140, 142, 144, 216, 218, 220, 223, 225}
// #define FRONT_LASERS_INDICIES {354, 356, 357, 359, 361, 363, 364, 366, 368}
// #define REAR_LASERS_INDICES   {1, 3, 4, 6, 8, 9, 11, 13, 14}
#define FRONT_LASERS_INDICIES {359, 361, 363}
#define REAR_LASERS_INDICES   {6, 8, 9}
#define LEFT_LASERS_INDICIES  {454, 455, 456}
#define RIGHT_LASERS_INDICIES {255, 257, 258}

namespace pose_follower{
  WallFollower::WallFollower(){}

  void WallFollower::initialize(){
    ros::NodeHandle nh;
    std::string scan_topic;
    distance_threshold_ = 0.7;
    ros::param::param<std::string>("~scan_topic", scan_topic, "/scan_unified");
    ros::param::param<double>("~min_obstacle_distance", distance_threshold_, distance_threshold_);
    // ros::param::param<std::string>("~scan_rear", scan_rear_topic, "/scan_rear");

    front_scan_sub_ = nh.subscribe(scan_topic, 1, &WallFollower::callbackLaserScan, this);

    left_mean_ = 0.0; right_mean_ = 0.0;
    front_mean_ = 0.0; rear_mean_ = 0.0;

  }

  bool WallFollower::is_obstacle_in_range(){
    if(front_mean_ < distance_threshold_ || rear_mean_ < distance_threshold_){
      return true;
    }
    return false;
  }

  void WallFollower::callbackLaserScan(const sensor_msgs::LaserScan& scan){
    ROS_DEBUG_STREAM("Scan Count : " << scan.ranges.size());
    // double ranges_double[scan.ranges.size()];
    double scan_range;
    unsigned int left_indices[SIGNIFACANT_LASER_COUNT] = LEFT_LASERS_INDICIES;
    unsigned int right_indices[SIGNIFACANT_LASER_COUNT] = RIGHT_LASERS_INDICIES;
    unsigned int front_indices[SIGNIFACANT_LASER_COUNT] = FRONT_LASERS_INDICIES;
    unsigned int rear_indices[SIGNIFACANT_LASER_COUNT] = REAR_LASERS_INDICES;
    double left_sum = 0.0, right_sum = 0.0, front_sum = 0.0, rear_sum = 0.0;
    // std::vector<unsigned int> range_inds;
    std::stringstream op_str;
    // op_str << "Indeices < 1.0 : ";
    // range_inds.clear();
    for(unsigned int i = 0; i < scan.ranges.size(); ++i){
      if(scan.ranges[i] < scan.range_min)
        scan_range = (double)scan.range_max;
      else
        scan_range = (double)scan.ranges[i];

      // if(scan_range < 0.5)
      //   op_str << i << ", ";
      for(unsigned int j=0; j < SIGNIFACANT_LASER_COUNT; ++j){

        if(i == left_indices[j])
          left_sum += scan_range;
        else if(i == right_indices[j])
          right_sum += scan_range;
        else if(i == front_indices[j])
          front_sum += scan_range;
        else if( i == rear_indices[j])
          rear_sum += scan_range;

      }
    }
    left_mean_ = left_sum / (double)SIGNIFACANT_LASER_COUNT;
    right_mean_ = right_sum / (double)SIGNIFACANT_LASER_COUNT;
    front_mean_ = front_sum / (double)SIGNIFACANT_LASER_COUNT;
    rear_mean_ = rear_sum / (double)SIGNIFACANT_LASER_COUNT;
    // ROS_DEBUG_STREAM("Leaser Readings: (" << left_mean_ << ", "
    //                                       << right_mean_ << ", "
    //                                       << front_mean_ << ", "
    //                                       << rear_mean_ << ")");
    // ROS_DEBUG_STREAM(op_str.str());
  }

};
