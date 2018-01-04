#!/home/dna14/anaconda3/envs/deep_q/bin/python
# /usr/bin/env python


import rospy
from rospy import logdebug, loginfo
import numpy as np
from dqn.dqn import DeepQAgent
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
        # self.cnn_input_count = (0,0,0)


        self.cnn_input_count = (self.inscribed_circle_frontier_radius * 2, self.inscribed_circle_frontier_radius * 2, self.cluster_count)# frontier areas.
        self.nn_input_count = 2 # robot pose.
        self.nn_input_count += 2 * self.cluster_count # frontier poses.
        self.nn_input_count += self.laser_readings_count # laser readings.
        self.nn_input_count = (self.nn_input_count,)
        rospy.loginfo("cnn_input_count : %s", str(self.cnn_input_count))
        self.agent= DeepQAgent(state_size=[self.cnn_input_count, self.nn_input_count],
                               number_of_actions=self.cluster_count,
                               learning_rate=learning_rate,
                               discount=discount_factor,
                               mbsz=10,
                               save_freq=1,
                               random_episodes=2000,
                               cnn_layers=[[20,8,5,'relu'],[20,6,2,'relu'],[200,activation]],
                               nn_layers=[[100, 'relu']])
        self.agent.log["debug"] = rospy.logdebug
        self.agent.log["info"] = rospy.loginfo
        self.last_state = None
        self.last_action = -1;
        self.agent.model.summary()
        self.rand_pub = rospy.Publisher("random_action", Bool, queue_size=100)
        network_service = rospy.Service("dqn_service", DQNExplorationService, self.handel_dqn_calls)
        rospy.logdebug("Initialized!")

    def handel_dqn_calls(self, request):
        response = DQNExplorationServiceResponse()
        cnn_ip = np.zeros(self.cnn_input_count)
        nn_ip = tuple()
        action_index = -1
        for i,area in enumerate(request.areas):
            cnn_ip[:,:,i] = np.array(area.data).reshape((self.cnn_input_count[0], self.cnn_input_count[1]))

        for pose in request.frontierPoses:
            nn_ip += (pose.position.x, pose.position.y)
        nn_ip += (request.robotPose.position.x, request.robotPose.position.y)
        nn_ip += request.laserRanges
        state = [cnn_ip[None, :], np.array(nn_ip)[None, :]]
        logdebug("cnn input : \n%s", str(state[0].shape))
        logdebug("nn input : \n%s", str(state[1].shape))



        if self.last_state is None:
            rospy.logdebug("FOR THE FRIST TIME NO REWARD!")
            # self.agent.new_episode()
        else:
            rospy.logdebug("new sample with reward!")
            self.agent.end_episode(terminal_state=state)
            cost = self.agent.observe(request.lastReward)

        self.agent.new_episode()
        action_index, values = self.agent.act(state)

        msg = Bool()
        msg.data = self.agent.is_last_action_random
        self.rand_pub.publish(msg)
        rospy.logdebug("values are : %s", str(values))
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
