cmake_minimum_required(VERSION 2.8.3)
project(youbot_driver_ros_interface)


find_package(catkin REQUIRED COMPONENTS youbot_driver roscpp rosconsole actionlib tf control_msgs brics_actuator pr2_msgs diagnostic_msgs geometry_msgs nav_msgs pr2_msgs std_msgs std_srvs trajectory_msgs)
find_package(Boost REQUIRED COMPONENTS thread date_time filesystem system)

catkin_package(
	CATKIN_DEPENDS youbot_driver rosconsole roscpp pr2_msgs brics_actuator tf control_msgs diagnostic_msgs geometry_msgs nav_msgs pr2_msgs std_msgs std_srvs trajectory_msgs
)

## preprocessor definitions
add_definitions(-DYOUBOT_CONFIGURATIONS_DIR=${youbot_driver_CONFIG_DIR})

## Set Compiler flags
include_directories(
	include
	${catkin_INCLUDE_DIRS}
	${Boost_INCLUDE_DIR}
)	


add_executable(youbot_driver_ros_interface 
    src/youbot_oodl.cpp 
    src/YouBotOODLWrapper.cpp
    src/YouBotConfiguration.cpp
    src/joint_state_observer_oodl.cpp
)

target_link_libraries(youbot_driver_ros_interface ${catkin_LIBRARIES} ${Boost_LIBRARIES})

add_dependencies(youbot_driver_ros_interface ${catkin_EXPORTED_TARGETS})

install(DIRECTORY launch/
	DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)


## run sudo command to enable direct network access
## disabled here by default because it is set by Debian postinst command

option(${PROJECT_NAME}_USE_SETCAP "Set permissions to access ethernet interface without sudo" ON)

set(SUDO_COMMAND gksudo)
if($ENV{USE_NORMAL_SUDO})
    set(SUDO_COMMAND sudo)
endif()

if(${PROJECT_NAME}_USE_SETCAP)
    add_custom_command(TARGET youbot_driver_ros_interface POST_BUILD
    COMMAND ${SUDO_COMMAND} setcap cap_net_raw+ep $<TARGET_FILE:youbot_driver_ros_interface>
)    
endif()


#test programs
add_executable(youbot_arm_test 
    src/examples/youbot_arm_test.cpp 
)
target_link_libraries(youbot_arm_test ${catkin_LIBRARIES} ${Boost_LIBRARIES} ${youbot_driver_LIBRARIES})

add_executable(youbot_2nd_arm_test 
	src/examples/youbot_2nd_arm_test.cpp 
)
target_link_libraries(youbot_2nd_arm_test ${catkin_LIBRARIES} ${Boost_LIBRARIES} ${youbot_driver_LIBRARIES})

install(TARGETS youbot_driver_ros_interface youbot_arm_test
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# enable python teleop script
catkin_python_setup()

install(PROGRAMS src/examples/youbot_keyboard_teleop.py
         DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
