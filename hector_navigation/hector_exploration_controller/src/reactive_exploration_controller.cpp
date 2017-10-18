//=================================================================================================
// Copyright (c) 2012, Stefan Kohlbrecher, TU Darmstadt
// All rights reserved.

// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//     * Neither the name of the Simulation, Systems Optimization and Robotics
//       group, TU Darmstadt nor the names of its contributors may be used to
//       endorse or promote products derived from this software without
//       specific prior written permission.

// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//=================================================================================================


#include <ros/ros.h>
#include <hector_nav_msgs/GetRobotTrajectory.h>

#include <ros/ros.h>
#include <move_base_msgs/MoveBaseAction.h>
#include <move_base_msgs/MoveBaseResult.h>
#include <actionlib/client/simple_action_client.h>

typedef actionlib::SimpleActionClient<move_base_msgs::MoveBaseAction> MoveBaseClient;

class ReactiveExplorationController
{
public:
  ReactiveExplorationController():
  move_base_client_("move_base",false)
  {
    ros::NodeHandle nh;

    // while(!move_base_client_.waitForServer(ros::Duration(5.0))){
    //   ROS_INFO("Waiting for the move_base action server to come up");
    // }
    first_time_ = true;
    goal_id_ = 0;
    exploration_plan_generation_timer_ = nh.createTimer(ros::Duration(1.0), &ReactiveExplorationController::timerPlanExploration, this, false );
    exploration_plan_service_client_ = nh.serviceClient<hector_nav_msgs::GetRobotTrajectory>("get_exploration_path");
    first_aborted_goal_id = -1;
    // path_follower_.initialize(&tfl_);
    // cmd_vel_generator_timer_ = nh.createTimer(ros::Duration(0.1), &ReactiveExplorationController::timerCmdVelGeneration, this, false );
    // vel_pub_ = nh.advertise<geometry_msgs::Twist>("cmd_vel", 10);

  }

  void timerPlanExploration(const ros::TimerEvent& e)
  {
    hector_nav_msgs::GetRobotTrajectory srv_exploration_plan;
    move_base_msgs::MoveBaseGoal goal;
    ROS_DEBUG_STREAM("move base state rejected: " << move_base_client_.getState().toString());
    if(move_base_client_.getState() == actionlib::SimpleClientGoalState::SUCCEEDED ||
       move_base_client_.getState() == actionlib::SimpleClientGoalState::ABORTED ||
       first_time_){
        first_time_ = false;
        if(move_base_client_.getState() == actionlib::SimpleClientGoalState::ABORTED &&
           first_aborted_goal_id < 0){
             first_aborted_goal_id = goal_id_ - 1;
        }else if(move_base_client_.getState() == actionlib::SimpleClientGoalState::ABORTED &&
                 first_aborted_goal_id >= 0 &&
                 (goal_id_ - first_aborted_goal_id > 10)){
                   first_aborted_goal_id = -1;
                   ROS_INFO("-----------------------------------SUHTING DOWN NODE -------------------");
                   ros::shutdown();
        }else if(move_base_client_.getState() == actionlib::SimpleClientGoalState::SUCCEEDED){
          first_aborted_goal_id = -1;
        }
        if (exploration_plan_service_client_.call(srv_exploration_plan)){
          unsigned int arr_size = (unsigned int)srv_exploration_plan.response.trajectory.poses.size();
          ROS_INFO("Generated exploration path with %u poses", arr_size);
          if(arr_size <= 0){
            // ROS_INFO("-----------------------------------SUHTING DOWN NODE -------------------");
            // ros::shutdown();
            first_time_ = true;
            return;
          }
          goal.target_pose = srv_exploration_plan.response.trajectory.poses[arr_size -1];
          ROS_DEBUG_STREAM("goal frame id : " << goal.target_pose.header.frame_id);
          goal.target_pose.header.seq = goal_id_;
          ROS_DEBUG_STREAM("goal sequence # : " << goal.target_pose.header.seq);
          ++goal_id_;
          goal.target_pose.header.stamp = ros::Time::now();
          // ROS_DEBUG_STREAM("target_pose header stamp : " << goal.target_pose.header.stamp);

          move_base_client_.sendGoal(goal);
        }else{
          ROS_WARN("Service call for exploration service failed");
        }
    }else{
      ROS_DEBUG("Still Executing Plan!");
    }
  }

  // void timerCmdVelGeneration(const ros::TimerEvent& e)
  // {
  //   geometry_msgs::Twist twist;
  //   path_follower_.computeVelocityCommands(twist);
  //   vel_pub_.publish(twist);
  // }


protected:
  ros::ServiceClient exploration_plan_service_client_;
  // ros::Publisher vel_pub_;

  // tf::TransformListener tfl_;

  // pose_follower::HectorPathFollower path_follower_;

  ros::Timer exploration_plan_generation_timer_;
  // ros::Timer cmd_vel_generator_timer_;
  MoveBaseClient move_base_client_;
  bool first_time_;
  unsigned int goal_id_;
  int first_aborted_goal_id;
};

int main(int argc, char **argv) {
  ros::init(argc, argv, ROS_PACKAGE_NAME);

  ReactiveExplorationController ec;

  ros::spin();

  return 0;
}
