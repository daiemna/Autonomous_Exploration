cmake_minimum_required(VERSION 2.8.3)
project(youbot_battery_monitor)

find_package(catkin REQUIRED
  COMPONENTS
    diagnostic_msgs
    pr2_msgs
    roscpp
    roslint
)

catkin_package(
  CATKIN_DEPENDS 
    diagnostic_msgs
    pr2_msgs
)

include_directories(
  ${catkin_INCLUDE_DIRS}
  include
)


### EXECUTABLES
add_executable(youbot_battery_monitor
  src/youbot_battery_monitor_node.cpp
  src/youbot_battery_monitor.cpp 
)
add_dependencies(youbot_battery_monitor ${catkin_EXPORTED_TARGETS})
target_link_libraries(youbot_battery_monitor 
  ${catkin_LIBRARIES}
  stdc++
)


### TESTS
roslint_cpp()


### INSTALLS
install(TARGETS youbot_battery_monitor
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
  FILES_MATCHING PATTERN "*.h"
)

install(FILES youbot_battery_monitor.sh
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
