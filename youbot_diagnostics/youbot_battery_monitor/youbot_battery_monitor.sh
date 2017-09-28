#!/bin/bash

if [ ! $ROS_MASTER_URI ]; then
    export ROS_MASTER_URI=http://localhost:11311
fi

/etc/youbot/youbot_battery_monitor /dev/youbot/lcd_display eth1 wlan0 &
