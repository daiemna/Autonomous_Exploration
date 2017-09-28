/*
 * Copyright 2016 Bonn-Rhein-Sieg University
 *
 * Author: Santosh Thoduka
 * Based on code by: Sergey Alexandrov
 *
 */

#include <ros/ros.h>
#include <tf/transform_listener.h>

#include <pcl_conversions/pcl_conversions.h>

#include <pcl/PCLPointCloud2.h>
#include <pcl/conversions.h>
#include <pcl/common/common.h>

#include "mcr_scene_segmentation/aliases.h"
#include "mcr_scene_segmentation/cloud_accumulation.h"
#include <mcr_scene_segmentation/cloud_accumulator.h>

using mcr_scene_segmentation::CloudAccumulatorNode;

CloudAccumulatorNode::CloudAccumulatorNode() :
    octree_resolution_(0.05), event_based_termination_(true),
    clouds_to_accumulate_(1), current_cloud_count_(0),
    publish_period_(0.1), add_to_octree_(false), publish_accumulated_cloud_(false)
{
}

CloudAccumulatorNode::~CloudAccumulatorNode()
{
}

void CloudAccumulatorNode::onInit()
{
    NODELET_INFO("[CloudAccumulatorNode] CloudAccumulatorNode started");
    nh_ = getPrivateNodeHandle();

    nh_.param("octree_resolution", octree_resolution_, 0.05);
    nh_.param("event_based_termination", event_based_termination_, true);
    nh_.param("clouds_to_accumulate", clouds_to_accumulate_, 1);
    nh_.param("publish_period", publish_period_, 0.1);

    cloud_accumulation_ = CloudAccumulation::UPtr(new CloudAccumulation(octree_resolution_));

    // for publishing accumulated cloud
    timer_ = nh_.createTimer(ros::Duration(publish_period_), boost::bind(&CloudAccumulatorNode::timerCallback, this));
    timer_.stop();

    pub_accumulated_cloud_ = nh_.advertise<sensor_msgs::PointCloud2>("output", 1);
    pub_event_out_ = nh_.advertise<std_msgs::String>("event_out", 1);
    sub_event_in_ = nh_.subscribe("event_in", 1, &CloudAccumulatorNode::eventCallback, this);
}

void CloudAccumulatorNode::eventCallback(const std_msgs::String::ConstPtr &msg)
{
    std_msgs::String event_out;
    if (msg->data == "e_start")
    {
        sub_input_cloud_ = nh_.subscribe("input", 1, &CloudAccumulatorNode::pointcloudCallback, this);
        event_out.data = "e_started";
    }
    else if (msg->data == "e_add_cloud_start")
    {
        add_to_octree_ = true;
        // Not needed so that not to affect the action server
        return;
    }
    else if (msg->data == "e_add_cloud_stop")
    {
        add_to_octree_ = false;
        event_out.data = "e_add_cloud_stopped";
    }
    else if (msg->data == "e_start_publish")
    {
        timer_.start();
        event_out.data = "e_started_publish";
    }
    else if (msg->data == "e_stop_publish")
    {
        timer_.stop();
        event_out.data = "e_stopped_publish";
    }
    else if (msg->data == "e_publish")
    {
        if (cloud_accumulation_->getCloudCount() > 0)
        {
            sensor_msgs::PointCloud2 ros_cloud;
            PointCloud cloud;
            cloud.header.frame_id = frame_id_;
            cloud_accumulation_->getAccumulatedCloud(cloud);

            pcl::PCLPointCloud2 pc2;
            pcl::toPCLPointCloud2(cloud, pc2);
            pcl_conversions::fromPCL(pc2, ros_cloud);
            ros_cloud.header.stamp = ros::Time::now();
            pub_accumulated_cloud_.publish(ros_cloud);
            event_out.data = "e_done";
        }
    }
    else if (msg->data == "e_reset")
    {
        cloud_accumulation_->reset();
        event_out.data = "e_reset";
    }
    else if (msg->data == "e_stop")
    {
        cloud_accumulation_->reset();
        sub_input_cloud_.shutdown();
        timer_.stop();
        event_out.data = "e_stopped";
    }
    else
    {
        return;
    }
    pub_event_out_.publish(event_out);
}

void CloudAccumulatorNode::pointcloudCallback(const sensor_msgs::PointCloud2::ConstPtr &msg)
{
    PointCloud::Ptr cloud(new PointCloud);
    pcl::PCLPointCloud2 pc2;
    pcl_conversions::toPCL(*msg, pc2);
    pcl::fromPCLPointCloud2(pc2, *cloud);

    frame_id_ = msg->header.frame_id;

    if (add_to_octree_)
    {
        cloud_accumulation_->addCloud(cloud);
        current_cloud_count_++;
    }

    if (!event_based_termination_ && add_to_octree_)
    {
        if (current_cloud_count_ >= clouds_to_accumulate_)
        {
            std_msgs::String event_out;

            current_cloud_count_ = 0;
            add_to_octree_ = false;
            event_out.data = "e_add_cloud_stopped";
            pub_event_out_.publish(event_out);
        }
    }
}

void CloudAccumulatorNode::timerCallback()
{
    if (cloud_accumulation_->getCloudCount() > 0 && pub_accumulated_cloud_.getNumSubscribers() > 0)
    {
        sensor_msgs::PointCloud2 ros_cloud;
        PointCloud cloud;
        cloud.header.frame_id = frame_id_;
        cloud_accumulation_->getAccumulatedCloud(cloud);

        pcl::PCLPointCloud2 pc2;
        pcl::toPCLPointCloud2(cloud, pc2);
        pcl_conversions::fromPCL(pc2, ros_cloud);
        ros_cloud.header.stamp = ros::Time::now();
        pub_accumulated_cloud_.publish(ros_cloud);
    }
}
