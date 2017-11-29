# Set the base image
FROM ros:indigo-robot

# Dockerfile author / maintainer
MAINTAINER Daiem Ali <daiem.ali@smail.inf.h-brs.de>

# MAS repository
ADD ./ /catkin_ws/src/
RUN apt-get -y update && \
    apt-get install -y cmake g++ bash-completion build-essential git \
    ros-indigo-hector-map-tools ros-indigo-hector-nav-msgs && \
    HOME=/catkin_ws && \
    rosdep update

WORKDIR /catkin_ws

RUN /bin/bash -c 'source /opt/ros/indigo/setup.bash;catkin_init_workspace src/;cd src/mas_industrial_robotics;./repository.debs;cd /catkin_ws;catkin_make;catkin_make;echo "echo MAS_repository" >> /root/.bashrc;echo "source /opt/ros/indigo/setup.bash" >> /root/.bashrc;echo "source /catkin_ws/devel/setup.bash" >> /root/.bashrc'

ADD wait_for_it.sh /wait-for-it.sh
ADD as_entrypoint.sh /ros_entrypoint.sh
# ENTRYPOINT ["as_entrypoint.sh"]
