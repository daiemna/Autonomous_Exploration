/*
 * Copyright [2012] <Bonn-Rhein-Sieg University>
 *
 * youbot_battery_monitor.h
 *
 *  Created on: Nov 30, 2012
 *      Author: Frederik Hegger, Jan Paulus
 */

#include <youbot_battery_monitor/youbot_battery_monitor.h>

int main(int argc, char* argv[])
{
    ros::init(argc, argv, "youbot_battery_monitor");

    youbot::YoubotBatteryMonitor* yb_battery_monitor = new youbot::YoubotBatteryMonitor();

    if (argc != 4)
    {
        std::cout << "invalid arguments \n ./youbot_battery_monitor 'serial port' 'ethernet' 'wlan'" << std::endl;
        return 0;
    }

    // if connecting fails, retry every 2 seconds
    std::cout << "try to connect to serial port: " << argv[1] << std::endl;
    while (!yb_battery_monitor->connect(argv[1]))
    {
        sleep(2);
    }

    while (true)
    {
        yb_battery_monitor->publishStatusInformation(argv[2], argv[3]);
        sleep(2);
    }

    yb_battery_monitor->disconnect();

    return (0);
}

