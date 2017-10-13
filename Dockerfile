# Set the base image
FROM ros:indigo-robot

# Dockerfile author / maintainer
MAINTAINER Daiem Ali <daiem.ali@smail.inf.h-brs.de>

# MAS repository
ADD ./Autonomous_Exploration/ /catkin_ws/src/
RUN apt-get -y update
RUN apt-get install -y cmake g++ bash-completion build-essential git \
    ros-indigo-hector-map-tools ros-indigo-hector-nav-msgs

RUN HOME=/catkin_ws rosdep update

WORKDIR /catkin_ws

RUN /bin/bash -c '. /opt/ros/indigo/setup.bash;catkin_init_workspace src/'
RUN /bin/bash -c '. /opt/ros/indigo/setup.bash;cd src/mas_industrial_robotics;./repository.debs'
RUN /bin/bash -c '. /opt/ros/indigo/setup.bash;catkin_make;catkin_make'
# the following line is not working.
RUN echo "source /catkin_ws/devel/setup.bash" >> /root/.bashrc
