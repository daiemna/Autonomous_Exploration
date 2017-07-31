Setting Permissions on Executables
==================================

In order to run executables which use the youbot_driver without root, [setcap](http://linux.die.net/man/8/setcap) must be run. The provided postinst script will run setcap on the globally installed locations of the youbot_driver executables. This script is run automatically if you are using the ROS Debian packages.
