#!/usr/bin/env python

# import types
import rospy
from rospy import logdebug, loginfo
from utility.costmap import GlobalCostmap
from geometry_msgs.msg import Pose
import numpy as np
import tf
# import json
import datetime as dt
import os, os.path
# import subprocess

class ExplorationEvaluator:
    def save_data_to_csv(self, event):
        date = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        distance = self.distance_traveled
        area = self.gcm.get_known_map_area(in_meters=True)
        with open(self.csv_save_path, 'a') as myf:
            myf.write("%s;%f;%f\n"%(date, distance, area))
            logdebug("wrote : %s;%f;%f",date, distance, area)
        loginfo(os.system("/bin/bash -c 'rosrun map_server map_saver -f " +
                          os.path.join(self.map_path,"map_" +self.sim_time_string) +
                          "'"))
    # def call_save_map(self):
        # loginfo("hook called with path: %s",self.map_path + "map_"+self.sim_time_string);
        # return subprocess.check_output(["cd "+self.map_path+" && rosrun map_server map_save -f map_"])
        # loginfo(subprocess.check_output(["rosrun","map_server","map_saver","-f","map_"+self.sim_time_string]))
        # loginfo(os.system("/bin/bash -c 'rosrun map_server map_saver -f map_"+self.sim_time_string + "'"))
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
        # logdebug("robot distance traveled : {0}".format(self.distance_traveled))

    def __init__(self, name):

        is_debug_level = rospy.get_param("~debuging",True)
        # Initialize the node with rospy
        if is_debug_level:
            rospy.init_node(name, log_level=rospy.DEBUG)
        else:
            rospy.init_node(name)

        save_duration = rospy.get_param("~data_save_duration",1)
        self.csv_save_path = rospy.get_param("~save_path", 'csvs')
        self.distance_precesion = rospy.get_param("~distance_precesion",3)
        self.sim_time_string =  rospy.get_param("~sim_time_str",'none')
        self.map_path = rospy.get_param("~map_path", 'saved_map/')
        self.bag_path = rospy.get_param("~bag_path", 'saved_bags/')

        self.odom_pose = None
        self.tf_pose = None
        self.pre_pose = None
        self.distance_traveled = 0.0
        # self.tfl_ = tf.TransformListener(rospy.Duration.from_sec(1))
        self.gcm = GlobalCostmap("global_costmap","global_costmap_updates")
        # self.costmap = costmap_2d.Costmap2D("the_costmap", self.tfl)
        if not os.path.isdir(self.csv_save_path):
            logdebug("path does not exisit! %s", self.csv_save_path)
            os.mkdir(self.csv_save_path)
        if not os.path.isdir(self.map_path):
            logdebug("path does not exisit! %s", self.map_path)
            os.mkdir(self.map_path)
        if not os.path.isdir(self.bag_path):
            logdebug("path does not exisit! %s", self.bag_path)
            os.mkdir(self.bag_path)

        if self.sim_time_string == 'none':
            self.sim_time_string = dt.datetime.now().strftime("%d%m%Y_%H%M%S")
        self.csv_save_path = os.path.join(self.csv_save_path,
                                          'exploration_data_'+
                                          self.sim_time_string+
                                          '.csv')

        if not os.path.exists(self.csv_save_path):
            with open(self.csv_save_path, 'a') as myf:
                myf.write("timestamp;distance;area\n")
        logdebug("executing from : %s", os.getcwd())
        loginfo("saving csv at : %s", self.csv_save_path)
        # loginfo("namespace : %s", rospy.get_namespace())

        rospy.Subscriber("robot_pose", Pose, self.get_tf_pose_callback)
        rospy.Timer(rospy.Duration(save_duration), self.save_data_to_csv)
        # self.sim_start_time = rospy.Time.now()
        rosbag_cmd = "/bin/bash -c 'rosbag record -O "+os.path.join(self.bag_path,"rosbag_" +self.sim_time_string)+" "
        rosbag_cmd += " ".join(rospy.get_param("~recorded_topics",[])) +"'"
        rospy.logdebug("rosbag command : "+rosbag_cmd)
        os.system(rosbag_cmd)
        loginfo("%s started! "%(name))


if __name__ == '__main__':

    ee = ExplorationEvaluator('exploration_evaluator')
    # rospy.on_shutdown(ee.call_save_map)
    rospy.spin()
