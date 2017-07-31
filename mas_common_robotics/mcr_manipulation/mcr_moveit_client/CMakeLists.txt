cmake_minimum_required(VERSION 2.8.3)
project(mcr_moveit_client)

find_package(catkin REQUIRED
  COMPONENTS
    rospy
    roslint
)

catkin_python_setup()

catkin_package(
  CATKIN_DEPENDS
    rospy
    actionlib
    std_msgs
    geometry_msgs
    moveit_msgs
    brics_actuator
    moveit_commander
)

### TESTS
roslint_python()

roslint_add_test()

if(CATKIN_ENABLE_TESTING)
  find_package(roslaunch REQUIRED)
  find_package(rostest REQUIRED)

  roslaunch_add_file_check(ros/launch)
endif()

install(PROGRAMS
  ros/scripts/moveit_client_node
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)