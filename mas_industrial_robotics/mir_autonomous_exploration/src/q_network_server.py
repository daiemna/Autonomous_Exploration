#!/home/dna14/anaconda3/envs/ros_keras/bin/python
# /usr/bin/env python


import rospy
from rospy import logdebug, loginfo
import numpy as np
from rcnfq.rcnfq import NFQ
from mir_autonomous_exploration.srv import DQNExplorationService, DQNExplorationServiceResponse
from std_msgs.msg import Bool

class QAgentServer:
    """docstring for QAgentServer"""
    def __init__(self, name):
        is_debug_level = rospy.get_param("~debuging",True)
        # Initialize the node with rospy
        if is_debug_level:
            rospy.init_node(name, log_level=rospy.DEBUG)
        else:
            rospy.init_node(name)
        self.inscribed_circle_frontier_radius = rospy.get_param("~inscribed_circle_radius_area", 5)
        self.cluster_count = rospy.get_param("~max_frontier_count", 10)
        self.laser_readings_count = rospy.get_param("~max_laser_readings_count", 721)
        learning_rate = rospy.get_param("~model_parameters/learning_rate", 1)
        discount_factor = rospy.get_param("~model_parameters/discount_factor", 1)
        activation = rospy.get_param("~model_parameters/activation", 'sigmoid')
        rand_action = rospy.get_param("~model_parameters/epsilon-greedy", False)
        rospy.logdebug("model param : %s", str([activation,discount_factor, learning_rate, rand_action]))
        self.nn_input_count = 0

        self.nn_input_count += 2 * self.cluster_count # frontier poses.
        self.nn_input_count += (self.inscribed_circle_frontier_radius * 2)**2 * self.cluster_count # frontier areas.
        self.nn_input_count += 2 # robot pose.
        self.nn_input_count += self.laser_readings_count # laser readings.

        rospy.loginfo("nn_input_count : %d", self.nn_input_count)
        self.agent = NFQ(state_dim=self.nn_input_count,
                         nb_actions=self.cluster_count,
                         terminal_states=np.zeros((self.nn_input_count,)),
                         mlp_layers=[50,20],
                         discount_factor=discount_factor,lr=learning_rate,
                         activations=activation,
                         e_greedy=rand_action);
        self.last_state = None
        self.last_action = -1;

        self.rand_pub = rospy.Publisher("random_action", Bool, queue_size=100)
        networ_service = rospy.Service("dqn_service", DQNExplorationService, self.handel_dqn_calls)
        rospy.logdebug("Initialized!")

    def handel_dqn_calls(self, request):
        response = DQNExplorationServiceResponse()
        state = ()
        action_index = -1
        for pose in request.frontierPoses:
            state += (pose.position.x, pose.position.y)
        for area in request.areas:
            state += area.data
        state += (request.robotPose.position.x, request.robotPose.position.y)
        state += request.laserRanges
        state = np.array(state)
        logdebug("state : \n%s", str(state.shape))

        if self.last_state is None:
            rospy.logdebug("FOR THE FRIST TIME NO REWARD!")
        else:
            rospy.logdebug("new sample with reward!", )
            self.agent.fit_vectorized(self.last_state,
                                      self.last_action,
                                      request.lastReward,
                                      state,
                                      sample_count=1,
                                      validation=False)

        action_index = self.agent.greedy_action(state)
        msg = Bool()
        msg.data = self.agent.is_last_action_random
        self.rand_pub.publish(msg)
        rospy.logdebug("action index : %d", action_index)
        self.last_state = state
        self.last_action = action_index

        if action_index < 0:
            return None
        else:
            response.action = action_index
            return response


if __name__ == '__main__':
    qa = QAgentServer("q_network_server")

    rospy.spin()
    # pass
