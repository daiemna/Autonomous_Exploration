cmake_minimum_required(VERSION 2.8.3)
project(dmp)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  message_generation
)

find_package(cmake_modules REQUIRED)

find_package(Eigen REQUIRED)

add_message_files(
   FILES
   DMPData.msg
   DMPPoint.msg
   DMPTraj.msg
 )

add_service_files(
   FILES
   GetDMPPlan.srv
   LearnDMPFromDemo.srv
   SetActiveDMP.srv
 )

generate_messages()

catkin_package(
  INCLUDE_DIRS
    include
  LIBRARIES
    ${PROJECT_NAME}
  CATKIN_DEPENDS 
    common_msgs
    image_common
    message_runtime
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${Eigen_INCLUDE_DIRS}
)

add_library(${PROJECT_NAME}
  src/${PROJECT_NAME}.cpp
  src/fourier_approx.cpp
  src/radial_approx.cpp
  src/linear_approx.cpp)

add_executable(${PROJECT_NAME}_server
  nodes/dmp_server.cpp
)
add_dependencies(${PROJECT_NAME}
  ${PROJECT_NAME}_gencpp
)
target_link_libraries(${PROJECT_NAME}_server
  ${PROJECT_NAME}
  ${catkin_LIBRARIES}
  ${Eigen_LIBRARIES}
)

### INSTALLS
install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
  FILES_MATCHING PATTERN "*.h"
  PATTERN ".svn" EXCLUDE
)

install(TARGETS ${PROJECT_NAME}_server
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(TARGETS ${PROJECT_NAME}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
)