/*
 * @Author: Xinwei Chen
 * @Date: 2024-07-07 16:34:49
 * @Last Modified by: Xinwei Chen
 * @Last Modified time: 2024-07-07 16:34:51
 */

#include "trajectory/simulation.h"
#include <algorithm>
#include <ros/ros.h>

namespace trajectory
{
  namespace sim
  {

    nav_msgs::Odometry genOdomFromGZ(const gazebo_msgs::ModelStates::ConstPtr &gz_msg_ptr,
                                     const std::string &model_name)
    {
      nav_msgs::Odometry odom;
      odom.header.stamp = ros::Time::now();
      auto it = std::find(gz_msg_ptr->name.begin(), gz_msg_ptr->name.end(), model_name);
      if (it == gz_msg_ptr->name.end())
      {
        ROS_ERROR("[generateOdomFromGZ] Cannot find the target: %s", model_name.c_str());
        ros::shutdown();
        exit(1);
      }
      auto index = std::distance(gz_msg_ptr->name.begin(), it);
      odom.pose.pose = gz_msg_ptr->pose[index];
      odom.twist.twist = gz_msg_ptr->twist[index];
      return odom;
    }

  } // namespace sim
} // namespace trajectory