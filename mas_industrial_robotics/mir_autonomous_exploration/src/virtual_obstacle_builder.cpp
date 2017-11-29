#include <ros/ros.h>
#include <costmap_2d/costmap_2d.h>
#include <costmap_2d/costmap_2d_ros.h>

#include <std_msgs/UInt16.h>
#include <std_msgs/String.h>
#include <std_msgs/Header.h>

#include <geometry_msgs/PointStamped.h>
#include <geometry_msgs/PolygonStamped.h>
#include <geometry_msgs/Point.h>
#include <geometry_msgs/Point32.h>
#include <geometry_msgs/Polygon.h>

#include <nav_msgs/OccupancyGrid.h>

#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

#define MAX_OBSTACLE_POINTS 4
#define MIN_OBSTACLE_POINTS 2

class VirtualObstacleBuilder{
public:
  VirtualObstacleBuilder(){
    initialize();
    ROS_INFO("init Done!");
  }

  void initialize(){
    ros::NodeHandle nh;
    costmap_ros_ = new costmap_2d::Costmap2DROS("global_costmap", tfl_);

    mode_ = VOB_OBSTACLE_BUILDER;
    command_sub_ = nh.subscribe("vob_command",1,&VirtualObstacleBuilder::callbackVOBCommand,this);
    clicked_point_sub_ = nh.subscribe("clicked_point", 1, &VirtualObstacleBuilder::callbackClickedPoints,this);
    map_sub_ = nh.subscribe("map", 1, &VirtualObstacleBuilder::callbackMapCollecterEmitter, this);

    polygon_pub_ = nh.advertise<geometry_msgs::PolygonStamped>("vob_polygon", 100);
    obstacle_map_pub_ = nh.advertise<sensor_msgs::Image>("obstacle_map", 1);
    new_map_pub_ = nh.advertise<nav_msgs::OccupancyGrid>("new_map", 1);

    vob_points.clear();
    indexes_.clear();
  }

  void callbackVOBCommand(const std_msgs::UInt16& msg){
    ROS_DEBUG("callbackVOBCommand() %d", msg.data);
    if(msg.data == VOB_CLEAR_AREA){
      ROS_INFO("VobMode::VOB_CLEAR_AREA");
      mode_ = VOB_CLEAR_AREA;
    }else{
      ROS_INFO("VobMode::VOB_OBSTACLE_BUILDER");
      mode_ = VOB_OBSTACLE_BUILDER;
    }
  }

  void publish_map_image(cv::Mat& image){
    cv_bridge::CvImage img_bridge;
    sensor_msgs::Image img_msg;

    std_msgs::Header header;

    header.stamp = ros::Time::now();
    img_bridge = cv_bridge::CvImage(header, sensor_msgs::image_encodings::MONO8, image);
    img_bridge.toImageMsg(img_msg);
    obstacle_map_pub_.publish(img_msg);
    ROS_DEBUG_STREAM("Published image of map obstacle : " << image.cols << "x" << image.rows);
  }

  void publish_points_polygon(std::vector<geometry_msgs::Point>& points, std::string frame_id){
    geometry_msgs::PolygonStamped poly_msg;
    poly_msg.header.frame_id=frame_id;
    for(int i=0; i < points.size(); ++i){
        geometry_msgs::Point32 pt;
        pt.x = (float)points[i].x;
        pt.y = (float)points[i].y;
        pt.z = (float)points[i].z;
        poly_msg.polygon.points.push_back(pt);
        ROS_DEBUG("Published point (%.2f, %.2f, %.2f)", poly_msg.polygon.points[i].x, poly_msg.polygon.points[i].y, poly_msg.polygon.points[i].z);
    }

    polygon_pub_.publish(poly_msg);
    ROS_DEBUG("Published polygon!");
  }
  void save_points_to_polygon(const geometry_msgs::PointStamped& msg){
    if(vob_points.size() == 0 || vob_points.size() == MAX_OBSTACLE_POINTS){
      vob_points.clear();
    }
    vob_points.push_back(msg.point);
    if(vob_points.size() >= MIN_OBSTACLE_POINTS){
      ROS_DEBUG("Publishing Polygon!");
      publish_points_polygon(vob_points, msg.header.frame_id);
    }
    if(vob_points.size() == MAX_OBSTACLE_POINTS && mode_ == VOB_OBSTACLE_BUILDER){
      ROS_DEBUG("Saving Polygon as obstacle!");
      // saving polygon for later use.
      indexes_.push_back(vob_points);
    }else if(vob_points.size() == MAX_OBSTACLE_POINTS && mode_ == VOB_CLEAR_AREA){
      ROS_DEBUG("Saving Polygon for clearing!");
      eliminate_area_poly_ = vob_points;
    }
  }

  void callbackClickedPoints(const geometry_msgs::PointStamped& msg){
    ROS_DEBUG("callbackClickedPoints()");
    if(mode_ == VOB_OBSTACLE_BUILDER || mode_ == VOB_CLEAR_AREA){
      save_points_to_polygon(msg);
    }
  }


  void fill_image_with_poly(cv::Mat& image, std::vector<std::vector<geometry_msgs::Point> >& polygon_list, cv::Scalar area_color, bool cvt_color){
    unsigned int mx, my;
    cv::Mat points(MAX_OBSTACLE_POINTS,1,CV_32SC2);
    std::vector<geometry_msgs::Point> real_points;
    bool conv;
    for(unsigned int i=0; i < polygon_list.size(); ++i){
      ROS_DEBUG("-------------------polygon # %d---------------------",i+1);
      real_points = polygon_list[i];
      for(unsigned int j=0; j < MAX_OBSTACLE_POINTS; ++j){
        conv = costmap_ros_->getCostmap()->worldToMap((double)real_points[j].x, (double)real_points[j].y, my, mx);
        points.at<cv::Point>(j) = cv::Point(mx, my);
      }
      ROS_DEBUG_STREAM("map Points : \n" << points);
      cv::fillConvexPoly(image, points, area_color, 8);
    }
    if(cvt_color){
      cv::cvtColor(image, image,CV_RGB2GRAY);
      unsigned int ob_pix_count = 0;
      for(int i=0; i < image.rows; ++i){
        for(int j=0; j < image.cols; ++j){
          int pix_val = (int)image.at<unsigned char>(i, j);
          if(pix_val == 255){
            ob_pix_count += 1;
          }
        }
      }
      ROS_DEBUG("number of obstacle pixles : %d", ob_pix_count);
      cv::imwrite("poly.png", image);
    }

  }


  void callbackMapCollecterEmitter(const nav_msgs::OccupancyGrid& map_msg){
    ROS_DEBUG("callbackMapCollecter()");
    const int number_of_inds = indexes_.size();
    const int eliminate_area_inds = eliminate_area_poly_.size();
    ROS_DEBUG_STREAM("polygons : " << indexes_.size() << " obstacle_map_pub_ subs : " << obstacle_map_pub_.getNumSubscribers());
    cv::Mat map_image = cv::Mat::zeros(map_msg.info.width, map_msg.info.height,CV_8UC3);
    if(number_of_inds > 0){
      fill_image_with_poly(map_image, indexes_, cv::Scalar(255,255,255), false);
    }
    if(eliminate_area_inds == MAX_OBSTACLE_POINTS){
      std::vector<std::vector<geometry_msgs::Point> > eliminate_vector;
      eliminate_vector.push_back(eliminate_area_poly_);
      fill_image_with_poly(map_image, eliminate_vector, cv::Scalar(255,0,0), false);
    }
    nav_msgs::OccupancyGrid new_map_msg;
    new_map_msg.header = map_msg.header;
    new_map_msg.info = map_msg.info;
    new_map_msg.data = map_msg.data;
    if(number_of_inds > 0 || eliminate_area_inds > 0){
      unsigned int mx, my;
      if(obstacle_map_pub_.getNumSubscribers() > 0){
        publish_map_image(map_image);
      }
      ROS_DEBUG("converting to gray scale!");
      cv::cvtColor(map_image, map_image,CV_RGB2GRAY);
      for(int i=0; i < map_msg.data.size(); ++i){
        costmap_ros_->getCostmap()->indexToCells(i, mx, my);
        int pix_val = (int)map_image.at<unsigned char>(mx, my);
        if(pix_val == 255){
          new_map_msg.data[i] = 100;
        }else if(pix_val == 76){
          new_map_msg.data[i] = 0;
        }
      }
    }
    ROS_DEBUG("published new map!");
    new_map_pub_.publish(new_map_msg);
  }
private:

    enum VobMode {
      VOB_OBSTACLE_BUILDER = 0,
      VOB_CLEAR_AREA = 1
    }mode_;


protected:

  costmap_2d::Costmap2DROS* costmap_ros_;
  tf::TransformListener tfl_;

  ros::Publisher polygon_pub_;
  ros::Publisher obstacle_map_pub_;
  ros::Publisher new_map_pub_;

  ros::Subscriber command_sub_;
  ros::Subscriber clicked_point_sub_;
  ros::Subscriber map_sub_;

  std::vector<geometry_msgs::Point> vob_points;
  std::vector<std::vector<geometry_msgs::Point> > indexes_;
  std::vector<geometry_msgs::Point> eliminate_area_poly_;
};

int main(int argc, char **argv) {
  ros::init(argc, argv, ROS_PACKAGE_NAME);

  VirtualObstacleBuilder vob;

  ros::spin();

  return 0;
}
