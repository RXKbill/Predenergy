/*
 * @Author: Xinwei Chen
 * @Date: 2024-07-07 16:30:50
 * @Last Modified by: Xinwei Chen
 * @Last Modified time: 2024-07-07 16:30:47
 */

#ifndef TRAJECTORY_VISUAL_H
#define TRAJECTORY_VISUAL_H

#include <Eigen/Eigen>
#include <visualization_msgs/Marker.h>
#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/Path.h>
#include <ros/ros.h>

namespace trajectory
{
  namespace visual
  {

    enum class VISUAL_COLOR
    {
      RED,
      GREEN,
      BLUE,
    };

    /**
     * @brief Generate a visualization marker for Rviz
     *
     * @param start start point
     * @param end end point
     * @param color RED or GREEN
     * @param id unique id
     * @param frame_id
     * @param ns
     * @return visualization_msgs::Marker
     */
    visualization_msgs::Marker genVisualMarker(const Eigen::Vector3d &start,
                                               const Eigen::Vector3d &end,
                                               const VISUAL_COLOR &color,
                                               int id,
                                               const std::string &frame_id = "world",
                                               const std::string &ns = "my_namespace");

    struct TrajectoryVisualizer
    {
    public:
      TrajectoryVisualizer() = delete;
      explicit TrajectoryVisualizer(const std::string &topic_name,
                                    const ros::NodeHandle &nh) : nh_(nh)
      {
        trajectory_pub_ = nh_.advertise<nav_msgs::Path>(topic_name, 1);
      }
      TrajectoryVisualizer(const TrajectoryVisualizer &) = delete;
      TrajectoryVisualizer &operator=(const TrajectoryVisualizer &) = delete;
      TrajectoryVisualizer(TrajectoryVisualizer &&) = delete;
      TrajectoryVisualizer &operator=(TrajectoryVisualizer &&) = delete;
      virtual ~TrajectoryVisualizer() = default;

      template <typename T>
      void setTrajectory(const T &trajectory, const std::string &frame_id = "world")
      {
        current_path_.header.frame_id = frame_id;
        current_path_.header.stamp = ros::Time::now();
        current_path_.poses.clear();
        // assume T has member points
        current_path_.poses.resize(trajectory.points.size());
        for (size_t i = 0; i < trajectory.points.size(); i++)
        {
          geometry_msgs::PoseStamped temp;
          temp.pose.position.x = trajectory.points[i].position.x();
          temp.pose.position.y = trajectory.points[i].position.y();
          temp.pose.position.z = trajectory.points[i].position.z();
          temp.pose.orientation.w = trajectory.points[i].orientation.w();
          temp.pose.orientation.x = trajectory.points[i].orientation.x();
          temp.pose.orientation.y = trajectory.points[i].orientation.y();
          temp.pose.orientation.z = trajectory.points[i].orientation.z();
          current_path_.poses[i] = temp;
        }
      }

      void publishTrajectory()
      {
        trajectory_pub_.publish(current_path_);
      }

    private:
      ros::NodeHandle nh_;
      ros::Publisher trajectory_pub_;
      nav_msgs::Path current_path_;
    }; // struct TrajectoryVisualizer

  } // namespace visual
} // namespace trajectory

#endif // TRAJECTORY_VISUAL_H