#include <ros/ros.h>
#include <costmap_2d/costmap_2d.h>
#include <costmap_2d/costmap_2d_ros.h>

#include <mir_autonomous_exploration/FrontierArea.h>
#include <mir_autonomous_exploration/GetRobotDestination.h>
#include <mir_autonomous_exploration/DQNExplorationService.h>
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/PoseArray.h>
#include <geometry_msgs/PoseStamped.h>
#include <sensor_msgs/LaserScan.h>
#include <std_msgs/Float32.h>

#include <boost/shared_array.hpp>
#include <math.h>

#include <opencv2/opencv.hpp>

class QExplorationPlanner{
public:
  QExplorationPlanner():
  costmap_ros_(0),
  costmap_(0),
  map_width_(0),
  map_height_(0){
    initalize();
    ROS_INFO("init done!");
  }

  void initalize(){
    // ros::NodeHandle nh;
    ros::NodeHandle pnh("~/");
    ros::NodeHandle nh;
    is_first_ = true;

    costmap_ros_ = new costmap_2d::Costmap2DROS("global_costmap", tfl_);
    costmap_ = costmap_ros_->getCostmap();

    pnh.param<double>("goal_reached_dist", p_goal_reached_dist_, 0.5);
    pnh.param<int>("max_frontier_count", p_cluster_count_, 10);
    pnh.param<bool>("use_inflated_obstacles", p_use_inflated_obs_, false);
    pnh.param<int>("inscribed_circle_radius_area", p_inscribed_circle_frontier_radius_, 5);
    nh.param<double>("sensor_range", p_sensor_range_, 4.5);
    robot_mileage_ = -1.0;
    setupMap();
    destination_server_ = pnh.advertiseService("robot_destination",
                                              &QExplorationPlanner::qlearned_destination_callback,
                                              this);
    deep_q_network_client_ = nh.serviceClient<mir_autonomous_exploration::DQNExplorationService>("dqn_service");

    robot_pose_sub_ = nh.subscribe("robot_pose", 1, &QExplorationPlanner::robot_pose_cb, this);
    laser_scan_sub_ = nh.subscribe("scan_unified", 1, &QExplorationPlanner::laser_scan_cb, this);

  }

  bool qlearned_destination_callback(mir_autonomous_exploration::GetRobotDestination::Request &req,
                                     mir_autonomous_exploration::GetRobotDestination::Response &res){
      // res.destination.header.frame_id = "map";
      ROS_DEBUG("New Request for Destination!");
      setupMap();
      float known_area = get_map_explored_area_cells();

      mir_autonomous_exploration::DQNExplorationService exploration_srv;
      std::string global_frame = costmap_ros_->getGlobalFrameID();
      exploration_srv.request.header.frame_id = global_frame;
      if(is_first_){
        ROS_DEBUG("ITS for the first time!");
        last_robot_mileage_ = robot_mileage_;
        last_know_area_ = known_area;
        is_first_ = false;
      }

      float distance_travled = robot_mileage_ - last_robot_mileage_;

      unsigned int sensor_range_cell = costmap_->cellDistance(p_sensor_range_);
      ROS_DEBUG("sensor_range_cell : %u", sensor_range_cell);

      unsigned int dist_cell = costmap_->cellDistance((double)distance_travled);
      ROS_DEBUG("distance travled %u", dist_cell);

      float area_covered = (sensor_range_cell * dist_cell) +
                           (M_PI * sensor_range_cell*sensor_range_cell);
      ROS_DEBUG("area_covered : %f", area_covered);
      // rewad = (known_area - last_know_area_)/area_covered + 1/distance_travled;
      ROS_DEBUG("known_area %f", known_area);
      ROS_DEBUG("last_know_area_ %f", last_know_area_);
      ROS_DEBUG("know area difference %f", (known_area - last_know_area_));
      exploration_srv.request.lastReward = 0.0;

      if((known_area - last_know_area_) > 0.0){
          exploration_srv.request.lastReward += (known_area - last_know_area_)/area_covered;
      }
      if(dist_cell > 2){
          exploration_srv.request.lastReward += 1/dist_cell;
      }


      ROS_DEBUG("Reward accumlated : %f", exploration_srv.request.lastReward);

      if(!findFrontiers(exploration_srv.request.frontierPoses)){
        ROS_ERROR("Frontiers not found!");
        return false;
      }
      get_frontier_area(exploration_srv.request.areas, exploration_srv.request.frontierPoses);

      exploration_srv.request.frontierCount = p_cluster_count_;

      exploration_srv.request.robotPose = robot_pose_;
      exploration_srv.request.laserRanges = laser_ranges_;

      // call the q_network service.
      // TODO: SEND THE LAST REWARD use example of exploration_evaluator.
      // TODO: distance should be normalized, but with what?
      // rewad = area_explored/area_coverd + 1/distance_travled
      if(deep_q_network_client_.call(exploration_srv)){
        int frontier_index = exploration_srv.response.action;
        res.destination.header.frame_id = costmap_ros_->getGlobalFrameID();
        res.destination.pose = exploration_srv.request.frontierPoses[frontier_index];
        last_robot_mileage_ = robot_mileage_;
        last_know_area_ = known_area;
        return true;
      }

      // last_robot_mileage_ = robot_mileage_;
      // last_know_area_ = known_area;
      return false;
  }

  void laser_scan_cb(sensor_msgs::LaserScan msg){
    laser_ranges_ = msg.ranges;
    // ROS_DEBUG("number of laser_readings %d", (int)laser_ranges_.size());
  }

  void robot_pose_cb(geometry_msgs::Pose msg){
    if(robot_mileage_ < 0){
      robot_pose_ = msg;
      robot_mileage_ = 0.0;
    }
    robot_mileage_ += sqrt(pow(msg.position.x - robot_pose_.position.x, 2) +
                             pow(msg.position.y - robot_pose_.position.y, 2));
    robot_pose_ = msg;
  }
  float get_map_explored_area_cells(){
    int known_cell_count = 0;
    for(int i = 0; i < num_map_cells_; i++){
      if(occupancy_grid_array_[i] != costmap_2d::NO_INFORMATION){
        ++known_cell_count;
      }
    }
    return (float)known_cell_count;
  }
  void get_frontier_area(std::vector<mir_autonomous_exploration::FrontierArea>& areas,
                         std::vector<geometry_msgs::Pose> frontiers){
    ROS_DEBUG("getting %d frontiers area",(int)frontiers.size());
    areas.clear();
    for(int i=0; i < frontiers.size(); ++i){
        mir_autonomous_exploration::FrontierArea area;
        area.data.clear();
        ROS_DEBUG("allocated area for %d", i);
        unsigned int mx,my;
        costmap_->worldToMap(frontiers[i].position.x, frontiers[i].position.y, mx, my);
        for(int x = mx - p_inscribed_circle_frontier_radius_;
            x < mx + p_inscribed_circle_frontier_radius_; ++x){
              for(int y = my - p_inscribed_circle_frontier_radius_;
                  y < my + p_inscribed_circle_frontier_radius_; ++y){
                    unsigned int index = x*y;
                    // ROS_DEBUG("index %u", index);
                    if(isValid(index)){
                        area.data.push_back((int)(occupancy_grid_array_[index] == costmap_2d::LETHAL_OBSTACLE ||
                                                   occupancy_grid_array_[index] == costmap_2d::FREE_SPACE));
                        // ROS_DEBUG("added value to area %c", occupancy_grid_array_[index]);
                    }else{
                        area.data.push_back((int)2);
                        ROS_DEBUG("index %u added value to invalid area %c", index, costmap_2d::NO_INFORMATION);
                    }
              }
        }
        ROS_DEBUG("Built Area size : %d", (int)area.data.size());
        areas.push_back(area);
      }
  }

  bool findFrontiers(std::vector<geometry_msgs::Pose> &frontiers){

    // get latest costmap
    clearFrontiers();

    // list of all frontiers in the occupancy grid
    std::vector<int> allFrontiers;
    frontiers.clear();

    // check if there is a subcriber to frontier publisher.
    // bool publishing_frontiers = (frontier_pub_.getNumSubscribers() > 0);
    bool publishing_frontiers = false;
    // ROS_DEBUG("frontier subs : %d", (int)frontier_pub_.getNumSubscribers());
    std::vector<geometry_msgs::Pose> fronts;

    // check for all cells in the occupancy grid whether or not they are frontier cells
    for(unsigned int i = 0; i < num_map_cells_; ++i){
      if(isFrontier(i)){
        if(p_cluster_count_ > 0){
          if(!isFrontierReached(i)){
            // ROS_DEBUG("Adding to allFrontiers!");
            allFrontiers.push_back(i);
          }
        }else{
            allFrontiers.push_back(i);
        }
      }
    }
    ROS_DEBUG("Found frontier : %d", (int)allFrontiers.size());
    if(p_cluster_count_ > 0 && allFrontiers.size() > p_cluster_count_){
      cv::Mat points(allFrontiers.size(), 1, CV_32FC2), labels, centers;
      int clusterCount = p_cluster_count_;
      for(int i = 0; i < allFrontiers.size(); ++i){
        double wx,wy;
        unsigned int mx,my;
        costmap_->indexToCells(allFrontiers[i], mx, my);
        costmap_->mapToWorld(mx,my,wx,wy);
        points.row(i) = cv::Scalar(wx, wy);
      }
      cv::kmeans(points,
                 clusterCount,
                 labels,
                 cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT,
                                  clusterCount * 10, 0.00001),
                 3, cv::KMEANS_PP_CENTERS, centers);
      ROS_DEBUG_STREAM("Cluster centers : " << centers);
      ROS_DEBUG_STREAM("labels count : " << labels.total() );
      ROS_DEBUG_STREAM("points count : " << points.rows );


      // char key;
      // std::cin >> key;

      for (unsigned int i=0; i < p_cluster_count_; ++i){

        geometry_msgs::Pose finalFrontier;
        getPoseFromWorld((double)centers.row(i).at<float>(0), (double)centers.row(i).at<float>(1), finalFrontier);
        // ROS_DEBUG("done calling getPoseFromWorld()");
        frontiers.push_back(finalFrontier);
        // ROS_DEBUG("done calling frontiers.push_back(finalFrontier)");
        if(publishing_frontiers){
          fronts.push_back(finalFrontier);
          ROS_DEBUG("done calling fronts.push_back(finalFrontier.pose)");
        }


      }
      ROS_DEBUG_STREAM("Cluster centers : \n" << centers);
      if(publishing_frontiers){
        geometry_msgs::PoseArray frontiers_msg;
        frontiers_msg.header.frame_id = costmap_ros_->getGlobalFrameID();
        frontiers_msg.poses = fronts;
        // frontier_pub_.publish(frontiers_msg);
        // ROS_DEBUG("Published %u frontiers!",(unsigned int)fronts.size());
      }
      // std::cin >> key;
      // if( key == 27 || key == 'q' || key == 'Q' ) // 'ESC'
      //     return false;
      // transmitClusterImage(points, labels, centers);
      ROS_DEBUG_STREAM("returning frontiers : " << frontiers.size());
      return (frontiers.size() > 0);
    }


    for(unsigned int i = 0; i < allFrontiers.size(); ++i){
      if(!isFrontierReached(allFrontiers[i])){
        geometry_msgs::Pose finalFrontier;
        getPoseFromIndex(allFrontiers[i], finalFrontier);
        frontiers.push_back(finalFrontier);

        if(publishing_frontiers){
          // front =
          fronts.push_back(finalFrontier);
        }


      }
      //}
    }
    if(publishing_frontiers){
      geometry_msgs::PoseArray frontiers_msg;
      frontiers_msg.header.frame_id = costmap_ros_->getGlobalFrameID();
      frontiers_msg.poses = fronts;
      // frontier_pub_.publish(frontiers_msg);
      // ROS_DEBUG("Published %u frontiers!",(unsigned int)fronts.size());
    }
    return (frontiers.size() > 0);
  }

  void getPoseFromIndex(int index, geometry_msgs::Pose &pose){
    double wx,wy;
    unsigned int mx,my;
    costmap_->indexToCells(index, mx, my);
    // costmap_->mapToWorld(mx,my,wx,wy);
    // std::string global_frame = costmap_ros_->getGlobalFrameID();
    // pose.header.frame_id = global_frame;
    pose.position.x = wx;
    pose.position.y = wy;
    pose.position.z = 0.0;

    // double yaw = getYawToUnknown(costmap_->getIndex(mx,my));
    double yaw = 0.0;

    //if(frontier_is_valid){

    pose.orientation = tf::createQuaternionMsgFromYaw(yaw);
  }

  void getPoseFromWorld(double wx, double wy, geometry_msgs::Pose &pose){
    // ROS_DEBUG("Called getPoseFromWorld()");
    unsigned int mx,my;

    // std::string global_frame = costmap_ros_->getGlobalFrameID();
    // pose.header.frame_id = global_frame;
    pose.position.x = wx;
    pose.position.y = wy;
    pose.position.z = 0.0;
    // ROS_DEBUG("Initializing Pose");
    // ROS_DEBUG_STREAM("Converting world : (" << wx << ", " << wy << ")" );
    // costmap_->worldToMap(wx, wy, mx, my);
    // ROS_DEBUG_STREAM("to map : (" << mx << ", " << my << ")" );
    // unsigned int the_ind =  costmap_->getIndex(mx,my);
    // ROS_DEBUG_STREAM("converted to index : " << the_ind );
    // double yaw = getYawToUnknown(the_ind);
    double yaw = 0.0;
    // ROS_DEBUG_STREAM("got Yaw : " << yaw);
    pose.orientation = tf::createQuaternionMsgFromYaw(yaw);
    // ROS_DEBUG("done getting yaw.");
  }

  void setupMap(){
    if ((map_width_ != costmap_->getSizeInCellsX()) || (map_height_ != costmap_->getSizeInCellsY())){
      map_width_ = costmap_->getSizeInCellsX();
      map_height_ = costmap_->getSizeInCellsY();
      num_map_cells_ = map_width_ * map_height_;

      // initialize exploration_trans_array_, obstacle_trans_array_, goalMap and frontier_map_array_
      // exploration_trans_array_.reset(new unsigned int[num_map_cells_]);
      // obstacle_trans_array_.reset(new unsigned int[num_map_cells_]);
      // is_goal_array_.reset(new bool[num_map_cells_]);
      frontier_map_array_.reset(new int[num_map_cells_]);
      // utility_trans_array_.reset(new float[num_map_cells_]);
      clearFrontiers();
      // resetMaps();
    }

    occupancy_grid_array_ = costmap_->getCharMap();
  }

  bool isFrontierReached(int point){

    tf::Stamped<tf::Pose> robotPose;
    if(!costmap_ros_->getRobotPose(robotPose)){
      ROS_WARN("[hector_exploration_planner]: Failed to get RobotPose");
    }
    geometry_msgs::PoseStamped robotPoseMsg;
    tf::poseStampedTFToMsg(robotPose, robotPoseMsg);

    unsigned int fx,fy;
    double wfx,wfy;
    costmap_->indexToCells(point,fx,fy);
    costmap_->mapToWorld(fx,fy,wfx,wfy);


    double dx = robotPoseMsg.pose.position.x - wfx;
    double dy = robotPoseMsg.pose.position.y - wfy;

    if ( (dx*dx) + (dy*dy) < (p_goal_reached_dist_ * p_goal_reached_dist_)) {
      // ROS_DEBUG("[hector_exploration_planner]: frontier is within the squared range of: %f", p_goal_reached_dist_);
      return true;
    }
    return false;

  }

  bool isFrontier(int point){
    const int MAX_INF_COUNT=2;
    const int ADJECENT_COUNT=8;
    if(isFreeFrontiers(point)){
      int adjacentPoints[ADJECENT_COUNT];
      getAdjacentPoints(point,adjacentPoints);
      for(int i = 0; i < ADJECENT_COUNT; ++i){
        if(isValid(adjacentPoints[i])){
          if(occupancy_grid_array_[adjacentPoints[i]] == costmap_2d::NO_INFORMATION){
            int no_inf_count = 0;
            int noInfPoints[ADJECENT_COUNT];
            getAdjacentPoints(adjacentPoints[i],noInfPoints);
            for(int j = 0; j < ADJECENT_COUNT; j++){
              if( isValid(noInfPoints[j]) && occupancy_grid_array_[noInfPoints[j]] == costmap_2d::NO_INFORMATION){
                ++no_inf_count;
                if(no_inf_count > MAX_INF_COUNT){
                  return true;
                }
              }
            }
          }
        }
      }
    }
    return false;
  }

  bool isFreeFrontiers(int point){

    if(isValid(point)){
      // if a point is not inscribed_inflated_obstacle, leathal_obstacle or no_information, its free


      if(p_use_inflated_obs_){
        if(occupancy_grid_array_[point] < costmap_2d::INSCRIBED_INFLATED_OBSTACLE){
          return true;
        }
      } else {
        if(occupancy_grid_array_[point] <= costmap_2d::INSCRIBED_INFLATED_OBSTACLE){
          return true;
        }
      }
    }
    return false;
  }

  void clearFrontiers(){
    std::fill_n(frontier_map_array_.get(), num_map_cells_, 0);
  }

  bool isValid(int point){
    return (point>=0) & (point < num_map_cells_);
  }

  void getAdjacentPoints(int point, int points[]){

    points[0] = left(point);
    points[1] = up(point);
    points[2] = right(point);
    points[3] = down(point);
    points[4] = upleft(point);
    points[5] = upright(point);
    points[6] = downright(point);
    points[7] = downleft(point);

  }

  int left(int point){
    // only go left if no index error and if current point is not already on the left boundary
    if((point % map_width_ != 0)){
      return point-1;
    }
    return -1;
  }
  int upleft(int point){
    if((point % map_width_ != 0) && (point >= (int)map_width_)){
      return point-1-map_width_;
    }
    return -1;

  }
  int up(int point){
    if(point >= (int)map_width_){
      return point-map_width_;
    }
    return -1;
  }
  int upright(int point){
    if((point >= (int)map_width_) && ((point + 1) % (int)map_width_ != 0)){
      return point-map_width_+1;
    }
    return -1;
  }
  int right(int point){
    if((point + 1) % map_width_ != 0){
      return point+1;
    }
    return -1;

  }
  int downright(int point){
    if(((point + 1) % map_width_ != 0) && ((point/map_width_) < (map_height_-1))){
      return point+map_width_+1;
    }
    return -1;

  }
  int down(int point){
    if((point/map_width_) < (map_height_-1)){
      return point+map_width_;
    }
    return -1;

  }
  int downleft(int point){
    if(((point/map_width_) < (map_height_-1)) && (point % map_width_ != 0)){
      return point+map_width_-1;
    }
    return -1;

  }
private:
  costmap_2d::Costmap2DROS* costmap_ros_;
  costmap_2d::Costmap2D* costmap_;
  tf::TransformListener tfl_;

  ros::ServiceServer destination_server_;
  ros::ServiceClient deep_q_network_client_;
  ros::Subscriber robot_pose_sub_;
  ros::Subscriber laser_scan_sub_;

  const unsigned char* occupancy_grid_array_;
  boost::shared_array<int> frontier_map_array_;

  geometry_msgs::Pose robot_pose_;
  std::vector<float> laser_ranges_;
  unsigned int num_map_cells_;
  unsigned int map_width_;
  unsigned int map_height_;
  bool is_first_;
  float last_know_area_;
  float robot_mileage_;
  float last_robot_mileage_;


  int p_cluster_count_;
  double p_goal_reached_dist_;
  bool p_use_inflated_obs_;
  int p_inscribed_circle_frontier_radius_;
  double p_sensor_range_;
};


int main(int argc, char **argv) {
  ros::init(argc, argv, ROS_PACKAGE_NAME);

  QExplorationPlanner qep;

  ros::spin();

  return 0;
}
