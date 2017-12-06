cmake_minimum_required(VERSION 2.8.3)
project(mir_autonomous_exploration)

find_package(catkin REQUIRED
  COMPONENTS
    actionlib
    std_msgs
    geometry_msgs
    map_msgs
    message_generation
    cv_bridge
    rospy
    roscpp
    rosconsole
    costmap_2d
)

find_package(OpenCV REQUIRED)

message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

catkin_python_setup()

add_message_files(
  DIRECTORY
    dqn_exploration_msgs/msg
  FILES
    FrontierArea.msg
)

add_service_files(
  DIRECTORY
    dqn_exploration_msgs/srv
  FILES
    DQNExplorationService.srv
    GetRobotDestination.srv
)

generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
  mir_autonomous_exploration
)

catkin_package(
 INCLUDE_DIRS include
 LIBRARIES
   ${PROJECT_NAME}
   virtual_obstacle_builder
 CATKIN_DEPENDS
   actionlib
   cv_bridge
   rospy
   roscpp
   rosconsole
   message_runtime
   std_msgs
   geometry_msgs
   map_msgs
   costmap_2d
 DEPENDS OpenCV
)

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)


add_executable(virtual_obstacle_builder src/virtual_obstacle_builder.cpp)
add_executable(q_exploration_planner src/q_exploration_planner.cpp)
add_executable(reactive_exploration_controller src/reactive_exploration_controller.cpp)

add_dependencies(q_exploration_planner ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
add_dependencies(reactive_exploration_controller ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

target_link_libraries(virtual_obstacle_builder ${catkin_LIBRARIES} ${OpenCV_LIBS})
target_link_libraries(q_exploration_planner ${catkin_LIBRARIES})
target_link_libraries(reactive_exploration_controller ${catkin_LIBRARIES})
