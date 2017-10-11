#ifndef HECTOR_WALL_FOLLOWER_H_
#define HECTOR_WALL_FOLLOWER_H_

#include <ros/ros.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Twist.h>
#include <sensor_msgs/LaserScan.h>

namespace pose_follower {
  class WallFollower{
  public:
    WallFollower();
    void initialize();
    bool is_obstacle_in_range();
    void callbackLaserScan(const sensor_msgs::LaserScan& scan);
    // void callbackRearScan(const sensor_msgs::LaserScan::ConstPtr& scan);

  private:
    ros::Subscriber front_scan_sub_;

    double left_mean_, right_mean_;
    double front_mean_, rear_mean_;

    double distance_threshold_;

    // ros::Subscriber rear_scan_sub_;
  };
};

#endif
