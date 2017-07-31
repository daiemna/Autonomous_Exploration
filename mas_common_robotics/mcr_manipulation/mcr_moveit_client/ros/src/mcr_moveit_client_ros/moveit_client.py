#!/usr/bin/env python
"""
This component moves a robotic manipulator, in a planned manner, to a specified
joint configuration using MoveIt!.

**Input(s):** target_configuration | target_string_pose | target_pose
** `target_configuration`: The joint configuration to which the manipulator
    will be moved.
    *type: brics_actuator.msg.JointPositions
  * `target_string_pose`: The string position to which the manipulator will
  be moved.
     *type: std_msgs.msg.String
  * `target_pose`: 3-D pose to which the manipulator will be moved.
     *type: geometry_msgs.msg.PoseStamped

**Parameter(s):**
  * `move_group`: MoveIt! interface.
  * `arm`: Name of the group to move.
  * `loop_rate`: Node cycle rate (in hz).

"""
# -*- encoding: utf-8 -*-

import rospy
import actionlib
import moveit_commander
import std_msgs.msg
import geometry_msgs.msg
import moveit_msgs.msg
import brics_actuator.msg


class MoveitClient(object):
    """
    Components that move the arm in a planned motion.

    """
    def __init__(self):
        # Params
        self.event = None
        self.target_configuration = None
        self.target_string_pose = None
        self.target_pose = None

        # MoveIt! interface
        move_group = rospy.get_param('~move_group', None)
        assert move_group is not None, "Move group must be specified."

        # Wait for MoveIt!
        client = actionlib.SimpleActionClient(move_group, moveit_msgs.msg.MoveGroupAction)
        rospy.loginfo("Waiting for '{0}' server".format(move_group))
        client.wait_for_server()
        rospy.loginfo("Found server '{0}'".format(move_group))

        # Name of the group to move
        arm = rospy.get_param('~arm', None)
        assert arm is not None, "The group to be moved must be specified (e.g. arm)."

        goal_position_tolerance = rospy.get_param("~goal_position_tolerance", 0.01)
        goal_orientation_tolerance = rospy.get_param("~goal_orientation_tolerance", 0.01)
        goal_joint_tolerance = rospy.get_param("~goal_joint_tolerance", 0.005)

        # Set up MoveIt!
        self.arm = moveit_commander.MoveGroupCommander(arm)

        # Setting up goal tolerences
        self.arm.set_goal_position_tolerance(goal_position_tolerance)
        self.arm.set_goal_orientation_tolerance(goal_orientation_tolerance)
        self.arm.set_goal_joint_tolerance(goal_joint_tolerance)

        # Whether MoveIt! should wait for the arm to be stopped.
        self.wait_for_motion = rospy.get_param('~wait_for_motion', True)

        # Node cycle rate (in hz)
        self.loop_rate = rospy.Rate(rospy.get_param('~loop_rate', 10))

        # Publishers
        self.event_out = rospy.Publisher("~event_out", std_msgs.msg.String, queue_size=1)

        # Subscribers
        rospy.Subscriber("~event_in", std_msgs.msg.String, self.event_in_cb)
        rospy.Subscriber(
            "~target_configuration", brics_actuator.msg.JointPositions,
            self.target_configuration_cb
        )
        rospy.Subscriber(
            "~target_string_pose", std_msgs.msg.String,
            self.target_string_pose_cb
        )
        rospy.Subscriber(
            "~target_pose", geometry_msgs.msg.PoseStamped,
            self.target_pose_cb
        )

    def target_configuration_cb(self, msg):
        """
        Obtains the joint configuration where the arm will be moved.

        """
        rospy.logdebug("Received target_configuration")
        self.target_configuration = msg

    def target_string_pose_cb(self, msg):
        """
        Obtains the string where the arm will be moved.

        """
        rospy.logdebug("Received target_string_pose: {0} ".format(msg.data))
        self.target_string_pose = msg

    def target_pose_cb(self, msg):
        """
        Obtains the pose where the arm will be moved.

        """
        rospy.logdebug("Received target pose")
        self.target_pose = msg

    def event_in_cb(self, msg):
        """
        Starts a planned motion based on the specified arm position.

        """
        rospy.logdebug("Received event: {0} ".format(msg.data))
        self.event = msg.data

    def start(self):
        """
        Starts the component.

        """
        rospy.loginfo("Ready to start...")
        state = 'INIT'

        while not rospy.is_shutdown():

            if state == 'INIT':
                state = self.init_state()
            elif state == 'IDLE':
                state = self.idle_state()
            elif state == 'RUNNING':
                state = self.running_state()

            rospy.logdebug("State: {0}".format(state))
            self.loop_rate.sleep()

    def init_state(self):
        """
        Executes the INIT state of the state machine.

        :return: The updated state.
        :rtype: str

        """
        if self.event == 'e_start':
            return 'IDLE'
        else:
            return 'INIT'

    def idle_state(self):
        """
        Executes the IDLE state of the state machine.

        :return: The updated state.
        :rtype: str

        """

        if self.event == 'e_stop':
            self.reset_component_data()
            self.event_out.publish('e_stopped')
            return 'INIT'
        elif self.target_configuration or self.target_string_pose or self.target_pose:
            return 'RUNNING'
        else:
            return 'IDLE'

    def running_state(self):
        """
        Executes the RUNNING state of the state machine.

        :return: The updated state.
        :rtype: str

        """
        if self.event == 'e_stop':
            self.arm.stop()
            self.reset_component_data()
            self.event_out.publish('e_stopped')
            return 'INIT'
        else:
            move_status = self.move_arm(self.wait_for_motion)
            if move_status:
                self.event_out.publish('e_success')
            else:
                self.event_out.publish('e_failure')

        self.reset_component_data()
        return 'INIT'

    def move_arm(self, wait=True):
        """
        Moves the arm to a specified joint configuration.

        :param wait: Wait for the execution of the trajectory to complete,
        :type wait: bool

        :return: False if the target configuration cannot be achieved.
        :rtype: bool

        """
        if self.target_configuration:
            joint_list = self.brics_joint_positions_to_list(self.target_configuration)
            try:
                self.arm.set_joint_value_target(joint_list)
            except Exception as e:
                rospy.logerr('unable to set target position: %s' % (str(e)))
                return False
        elif self.target_string_pose:
            try:
                self.arm.set_named_target(self.target_string_pose.data)
            except Exception as e:
                rospy.logerr('unable to set target position: %s' % (str(e)))
                return False
        elif self.target_pose:
            try:
                self.arm.set_pose_target(self.target_pose)
            except Exception as e:
                rospy.logerr('unable to set target position: %s' % (str(e)))
                return False
        status = self.arm.go(wait=wait)
        return status

    def reset_component_data(self):
        """
        Clears the data of the component.

        """
        self.event = None
        self.target_configuration = None
        self.target_string_pose = None
        self.target_pose = None

    def brics_joint_positions_to_list(self, joint_configuration):
        """
        Converts a BRICS message of joint positions into a list of real values.

        :param joint_configuration: Joint configuration as a BRICS message.
        :type joint_configuration: brics_actuator.msg.JointPositions

        :return: A list of real values representing the joints as specified
                 by the BRICS message.
        :rtype: list

        """
        return [joint.value for joint in joint_configuration.positions]


def main():
    rospy.init_node("moveit_client", anonymous=True)
    moveit_client = MoveitClient()
    moveit_client.start()
