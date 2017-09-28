/*
 * Copyright 2016 Bonn-Rhein-Sieg University
 *
 * Author: Santosh Thoduka
 * Based on code by: Sergey Alexandrov
 *
 */
#ifndef MCR_SCENE_SEGMENTATION_CLOUD_ACCUMULATOR_H
#define MCR_SCENE_SEGMENTATION_CLOUD_ACCUMULATOR_H

#include <string>
#include <nodelet/nodelet.h>
#include <pluginlib/class_list_macros.h>
#include <ros/ros.h>
#include <std_msgs/String.h>
#include <sensor_msgs/PointCloud2.h>

namespace mcr_scene_segmentation
{
class CloudAccumulatorNode : public nodelet::Nodelet
{
    public:
        CloudAccumulatorNode();
        virtual ~CloudAccumulatorNode();

    private:
        virtual void onInit();
        void eventCallback(const std_msgs::String::ConstPtr &msg);
        void pointcloudCallback(const sensor_msgs::PointCloud2::ConstPtr &msg);
        void timerCallback();


    private:
        ros::NodeHandle nh_;

        ros::Publisher pub_accumulated_cloud_;
        ros::Publisher pub_event_out_;

        ros::Subscriber sub_event_in_;
        ros::Subscriber sub_input_cloud_;

        ros::Timer timer_;

        CloudAccumulation::UPtr cloud_accumulation_;

        std::string frame_id_;
        double octree_resolution_;

        // if true cloud accumulation is stopped with event (e_stop)
        // if false, cloud accumulation is stopped when clouds_to_accumulate_ are accumulated
        bool event_based_termination_;
        // maximum clouds
        int clouds_to_accumulate_;
        int current_cloud_count_;

        double publish_period_;

        bool add_to_octree_;
        bool publish_accumulated_cloud_;
};

PLUGINLIB_DECLARE_CLASS(mcr_scene_segmentation,
    CloudAccumulatorNode, mcr_scene_segmentation::CloudAccumulatorNode, nodelet::Nodelet);
}  // namespace mcr_scene_segmentation
#endif  // MCR_SCENE_SEGMENTATION_CLOUD_ACCUMULATOR_H
