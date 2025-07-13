/*
 * @Author: Xinwei Chen 
 * @Date: 2024-07-07 16:30:42
 * @Last Modified by: Xinwei Chen
 * @Last Modified time: 2024-07-07 16:30:38
 */

#include "trajectory/visual.h"

namespace trajectory
{
  namespace visual
  {
    visualization_msgs::Marker genVisualMarker(const Eigen::Vector3d &start,
                                               const Eigen::Vector3d &end,
                                               const VISUAL_COLOR &color,
                                               int id,
                                               const std::string &frame_id,
                                               const std::string &ns)
    {
      visualization_msgs::Marker marker;
      marker.header.frame_id = frame_id;
      marker.header.stamp = ros::Time::now();
      marker.id = id;
      marker.ns = ns;
      marker.pose.orientation.w = 1.0;
      marker.color.a = 1.0;
      switch (color)
      {
      case VISUAL_COLOR::RED:
        marker.color.r = 1.0;
        break;
      case VISUAL_COLOR::GREEN:
        marker.color.g = 1.0;
        break;
      case VISUAL_COLOR::BLUE:
        marker.color.b = 1.0;
        break;
      default:
        break;
      }
      marker.scale.x = 0.1;
      marker.scale.y = 0.1;
      marker.scale.z = 0.1;
      marker.type = visualization_msgs::Marker::ARROW;
      geometry_msgs::Point start_point, end_point;
      start_point.x = start.x();
      start_point.y = start.y();
      start_point.z = start.z();
      end_point.x = end.x();
      end_point.y = end.y();
      end_point.z = end.z();
      marker.points.emplace_back(start_point);
      marker.points.emplace_back(end_point);
      return marker;
    }
  } // namespace visual
} // namespace trajectory