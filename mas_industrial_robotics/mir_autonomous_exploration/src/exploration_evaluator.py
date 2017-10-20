#!/usr/bin/env python

import rospy
from rospy import logdebug, loginfo
from utility import util
# from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose
import numpy as np

class ExplorationEvaluator:
    def get_odom_callback(self, odom):
        self.odom_pose = odom.pose.pose
        # logdebug("robot pose is : \n{0}".format(self.odom_pose.position))
    def get_tf_pose_callback(self, pose):
        self.tf_pose = pose
        if self.pre_pose is None:
            self.pre_pose = self.tf_pose

        c_pos = np.array([self.tf_pose.position.x, self.tf_pose.position.y])
        p_pos = np.array([self.pre_pose.position.x, self.pre_pose.position.y])

        distance = np.around(np.sqrt(np.sum((c_pos - p_pos)**2)),
                                            decimals=self.distance_precesion)

        self.distance_traveled += distance
        self.pre_pose = self.tf_pose
        logdebug("robot distance traveled : {0}".format(self.distance_traveled))
    def __init__(self, name):

        update_duration = rospy.get_param("data_update_duration",1.0)
        is_debug_level = rospy.get_param("log_level_debug",True)
        self.distance_precesion = rospy.get_param("distance_precesion",3)

        self.odom_pose = None
        self.tf_pose = None
        self.pre_pose = None
        self.distance_traveled = 0.0

        # Initialize the node with rospy
        if is_debug_level:
            rospy.init_node(name, log_level=rospy.DEBUG)
        else:
            rospy.init_node(name)
        loginfo("%s started()"%(name))

        # rospy.Subscriber("odom", Odometry, self.get_odom_callback)
        rospy.Subscriber("robot_pose", Pose, self.get_tf_pose_callback)
        # rospy.Timer(rospy.Duration.from_sec(pub_period),self.get_odom_callback)
        self.sim_start_time = rospy.Time.now()

if __name__ == '__main__':

    ExplorationEvaluator('exploration_evaluator')

    rospy.spin()
