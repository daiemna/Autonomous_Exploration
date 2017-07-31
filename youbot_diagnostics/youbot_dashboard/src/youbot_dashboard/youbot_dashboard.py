# Software License Agreement (BSD License)
#
# Copyright (c) 2012, Willow Garage, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#  * Neither the name of Willow Garage, Inc. nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import roslib
import rospy

from pr2_msgs.msg import PowerBoardState, DashboardState
import std_srvs.srv

from rqt_robot_dashboard.dashboard import Dashboard
from rqt_robot_dashboard.monitor_dash_widget import MonitorDashWidget
from rqt_robot_dashboard.console_dash_widget import ConsoleDashWidget

from python_qt_binding.QtCore import QSize
from python_qt_binding.QtGui import QMessageBox

from rqt_pr2_dashboard.pr2_battery import PR2Battery
from .youbot_motors import YoubotMotors
from .youbot_ethercat import YoubotEthercat

roslib.load_manifest('youbot_dashboard')


class YoubotDashboard(Dashboard):
    """
    Dashboard for the youBot

    :param context: the plugin context
    :type context: qt_gui.plugin.Plugin
    """
    def setup(self, context):
        self.name = 'Youbot Dashboard'
        self.max_icon_size = QSize(50, 30)
        self.message = None

        self._dashboard_message = None
        self._last_dashboard_message_time = 0.0

        self._raw_byte = None
        self.digital_outs = [0, 0, 0]

        self._console = ConsoleDashWidget(self.context, minimal=False)
        self._monitor = MonitorDashWidget(self.context)
        self._base_motors = YoubotMotors("base", self.on_base_motors_on_clicked, self.on_base_motors_off_clicked)
        self._arm_1_motors = YoubotMotors("arm", self.on_arm_1_motors_on_clicked, self.on_arm_1_motors_off_clicked)

        self._ethercat = YoubotEthercat('EtherCAT', self.on_reconnect_clicked)
        self._batteries = [PR2Battery(self.context)]

        self._dashboard_agg_sub = rospy.Subscriber('dashboard_agg', DashboardState, self.dashboard_callback)

    def get_widgets(self):
        return [[self._monitor, self._console], [self._base_motors, self._arm_1_motors], [self._ethercat],
                self._batteries]

    def check_motor_state(self, button_handle, component_name, msg, data_index):
        if (msg.power_board_state_valid and not msg.power_board_state.run_stop):
            if (msg.power_board_state.circuit_state[data_index] == PowerBoardState.STATE_ENABLED):
                button_handle.set_ok()
                button_handle.setToolTip(self.tr(component_name + " Motors: Switched ON"))
            elif (msg.power_board_state.circuit_state[data_index] == PowerBoardState.STATE_STANDBY):
                button_handle.set_warn()
                button_handle.setToolTip(self.tr(component_name + " Motors: Switched OFF"))
            elif (msg.power_board_state.circuit_state[data_index] == PowerBoardState.STATE_DISABLED):
                button_handle.set_error()
                button_handle.setToolTip(self.tr(component_name + " Motors: not connected"))
            else:
                button_handle.set_stale()
                button_handle.setToolTip(self.tr(component_name + " Motors: stale"))
        else:
            button_handle.set_stale()
            button_handle.setToolTip(self.tr(component_name + " Motors: stale"))

    def dashboard_callback(self, msg):
        """
        callback to process dashboard_agg messages

        :param msg: dashboard_agg DashboardState message
        :type msg: pr2_msgs.msg.DashboardState
        """
        self._dashboard_message = msg
        self._last_dashboard_message_time = rospy.get_time()

        # base and arm motors
        self.check_motor_state(self._base_motors, "Base", msg, 0)
        self.check_motor_state(self._arm_1_motors, "Arm 1", msg, 1)

        # battery
        if (msg.power_state_valid):
            self._batteries[0].set_power_state(msg.power_state)
        else:
            self._batteries[0].set_stale()

        # ethercat connection
        if (msg.power_board_state_valid):
            if not msg.power_board_state.run_stop:
                self._ethercat.set_connected()
                self._ethercat.setToolTip(self.tr("EtherCAT: connection established"))
            else:
                self._ethercat.set_disconnected()
                self._ethercat.setToolTip(self.tr("EtherCAT: connection lost"))
        else:
            self._ethercat.set_stale()
            self._ethercat.setToolTip(self.tr("EtherCAT: Stale"))

    def on_reconnect_clicked(self):
        if (self._dashboard_message is not None):
            reconnect = rospy.ServiceProxy("/reconnect", std_srvs.srv.Empty)

            try:
                reconnect()
            except rospy.ServiceException, e:
                QMessageBox.critical(self.ethercat,
                                     "Error", "Failed to reconnect the driver: service call failed with error: %s" %
                                     (e))

    def on_motors_clicked(self, button_handle, component_name, data_index, mode_name):
        if (self._dashboard_message is not None and self._dashboard_message.power_board_state_valid):

            # if mode == "Off"
            ok_state = PowerBoardState.STATE_ENABLED
            switch_state = PowerBoardState.STATE_STANDBY

            if (mode_name == "On"):
                ok_state = PowerBoardState.STATE_STANDBY
                switch_state = PowerBoardState.STATE_ENABLED

            if (self._dashboard_message.power_board_state.circuit_state[data_index] == ok_state):
                switch_motor_state = rospy.ServiceProxy("/" + component_name + "/switch" + mode_name + "Motors",
                                                        std_srvs.srv.Empty)

                try:
                    switch_motor_state()
                except rospy.ServiceException, e:
                    QMessageBox.critical(button_handle, "Error", "Failed to switch " + mode_name + " " +
                                         component_name + " motors: service call failed with error: %s" % (e))

            elif (self._dashboard_message.power_board_state.circuit_state[data_index] == switch_state):
                QMessageBox.critical(button_handle, "Error", component_name + " motors are already switched " +
                                     mode_name)
            elif (self._dashboard_message.power_board_state.circuit_state[data_index] ==
                  PowerBoardState.STATE_DISABLED):
                QMessageBox.critical(button_handle, "Error", component_name + " is not connected")

    def on_base_motors_on_clicked(self):
        self.on_motors_clicked(self._base_motors, "base", 0, "On")

    def on_base_motors_off_clicked(self):
        self.on_motors_clicked(self._base_motors, "base", 0, "Off")

    def on_arm_1_motors_on_clicked(self):
        self.on_motors_clicked(self._arm_1_motors, "arm_1", 1, "On")

    def on_arm_1_motors_off_clicked(self):
        self.on_motors_clicked(self._arm_1_motors, "arm_1", 1, "Off")

    def shutdown_dashboard(self):
        self._dashboard_agg_sub.unregister()

    def save_settings(self, plugin_settings, instance_settings):
        self._console.save_settings(plugin_settings, instance_settings)
        self._monitor.save_settings(plugin_settings, instance_settings)

    def restore_settings(self, plugin_settings, instance_settings):
        self._console.restore_settings(plugin_settings, instance_settings)
        self._monitor.restore_settings(plugin_settings, instance_settings)
