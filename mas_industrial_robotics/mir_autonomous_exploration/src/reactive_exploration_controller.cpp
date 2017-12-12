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
#include <mir_autonomous_exploration/GetRobotDestination.h>
#include <move_base_msgs/MoveBaseAction.h>
#include <move_base_msgs/MoveBaseResult.h>
#include <actionlib/client/simple_action_client.h>

typedef actionlib::SimpleActionClient<move_base_msgs::MoveBaseAction> MoveBaseClient;

class ReactiveExplorationController
{
public:
  ReactiveExplorationController():
  move_base_client_("move_base",true)
  {
    ros::NodeHandle nh;
    while(!move_base_client_.waitForServer(ros::Duration(1.0))){
      ROS_INFO("Waiting for the move_base action server to come up");
    }
    first_time_ = true;
    goal_id_ = 0;
    exploration_plan_generation_timer_ = nh.createTimer(ros::Duration(1.0), &ReactiveExplorationController::timerPlanExploration, this, false );

    exploration_goal_service_client_ = nh.serviceClient<mir_autonomous_exploration::GetRobotDestination>("/q_exploration_planner/robot_destination");
    first_aborted_goal_id = -1;
  }

  void timerPlanExploration(const ros::TimerEvent& e)
  {
    mir_autonomous_exploration::GetRobotDestination srv_exploration_destination;
    move_base_msgs::MoveBaseGoal goal;
    ROS_DEBUG_STREAM("move base state : " << move_base_client_.getState().toString());
    ROS_DEBUG_STREAM("is it first time : " << (first_time_?"YES":"NO"));
    // if(move_base_client_.getState() == actionlib::SimpleClientGoalState::PREEMPTED){
    //   ROS_DEBUG("waiting for move base's other goal to finish!");
    //   return;
    // }else{
    //   first_time_ = true;
    // }
    if(move_base_client_.getState() == actionlib::SimpleClientGoalState::SUCCEEDED ||
       move_base_client_.getState() == actionlib::SimpleClientGoalState::ABORTED ||
       first_time_){
        if(first_time_ || move_base_client_.getState() == actionlib::SimpleClientGoalState::PREEMPTED){
            srv_exploration_destination.request.last_goal_succeded = srv_exploration_destination.request.UNKNOWN;
        }else if(move_base_client_.getState() == actionlib::SimpleClientGoalState::SUCCEEDED){
            srv_exploration_destination.request.last_goal_succeded = srv_exploration_destination.request.YES;
        }else{
            srv_exploration_destination.request.last_goal_succeded = srv_exploration_destination.request.NO;
        }
        first_time_ = false;
        if(move_base_client_.getState() == actionlib::SimpleClientGoalState::ABORTED &&
           first_aborted_goal_id < 0){
             first_aborted_goal_id = goal_id_ - 1;
        }else if(move_base_client_.getState() == actionlib::SimpleClientGoalState::ABORTED &&
                 first_aborted_goal_id >= 0 &&
                 (goal_id_ - first_aborted_goal_id > 10)){
                   first_aborted_goal_id = -1;
                   ROS_INFO("-----------------------------------SUHTING DOWN NODE -------------------");
                  //  ros::shutdown();
        }else if(move_base_client_.getState() == actionlib::SimpleClientGoalState::SUCCEEDED){
          first_aborted_goal_id = -1;
        }
        if (exploration_goal_service_client_.call(srv_exploration_destination)){
          // ROS_DEBUG("RECIVED DESTINATION GOING TO SLEEP!");
          // ros::Duration(60).sleep();
          // return;

          ROS_INFO_STREAM("Generated exploration destination " << srv_exploration_destination.response.destination);
          goal.target_pose = srv_exploration_destination.response.destination;
          ROS_DEBUG_STREAM("goal frame id : " << goal.target_pose.header.frame_id);
          goal.target_pose.header.seq = goal_id_;
          ROS_DEBUG_STREAM("goal sequence # : " << goal.target_pose.header.seq);
          ++goal_id_;
          goal.target_pose.header.stamp = ros::Time::now();

          move_base_client_.sendGoal(goal);
        }else{
          ROS_WARN("Service call for exploration service failed!");
        }

    }else{
      ROS_DEBUG("Still Executing Plan!");
    }
  }

protected:
  ros::ServiceClient exploration_goal_service_client_;
  ros::Timer exploration_plan_generation_timer_;
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
