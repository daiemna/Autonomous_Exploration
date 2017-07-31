cmake_minimum_required(VERSION 2.8.3)
project(youbot_dashboard)

find_package(catkin REQUIRED
  COMPONENTS
    roslint
)

catkin_package()
catkin_python_setup()

### TESTS
roslint_python()

if(CATKIN_ENABLE_TESTING)
  find_package(roslaunch REQUIRED)

  roslaunch_add_file_check(launch)
endif()


### INSTALLS
install(FILES plugin.xml
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

install(PROGRAMS scripts/youbot_dashboard
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)

install(DIRECTORY images/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/images
)
