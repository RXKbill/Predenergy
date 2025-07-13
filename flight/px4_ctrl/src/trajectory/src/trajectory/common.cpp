/*
 * @Author: Xinwei Chen
 * @Date: 2024-07-07 18:28:24
 * @Last Modified by: Xinwei Chen
 * @Last Modified time: 2024-07-07 18:28:28
 */

// Reference: rpg_mpc (the original LICENSE is listed below)

/*    rpg_quadrotor_mpc
 *    A model predictive control implementation for quadrotors.
 *    Copyright (C) 2017-2018 Philipp Foehn,
 *    Robotics and Perception Group, University of Zurich
 *
 *    Intended to be used with rpg_quadrotor_control and rpg_quadrotor_common.
 *    https://github.com/uzh-rpg/rpg_quadrotor_control
 *
 *    This program is free software: you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation, either version 3 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#include "trajectory/common.h"

namespace trajectory
{
    namespace common
    {
        // Implementations

        TrajectoryPoint::TrajectoryPoint() : position(Eigen::Vector3d::Zero()),
                                             velocity(Eigen::Vector3d::Zero()),
                                             acceleration(Eigen::Vector3d::Zero()),
                                             jerk(Eigen::Vector3d::Zero()),
                                             orientation(Eigen::Quaterniond::Identity())
        {
        }

        TrajectoryPoint::TrajectoryPoint(const Eigen::Vector3d &p,
                                         const Eigen::Vector3d &v,
                                         const Eigen::Vector3d &a,
                                         const Eigen::Vector3d &j,
                                         const Eigen::Quaterniond &q) : position(p),
                                                                        velocity(v),
                                                                        acceleration(a),
                                                                        jerk(j),
                                                                        orientation(q)
        {
        }

        TrajectoryPoint::~TrajectoryPoint()
        {
        }



        Trajectory::Trajectory() : points()
        {
        }

        Trajectory::Trajectory(const trajectory::common::TrajectoryPoint &point) : points()
        {
            points.push_back(point);
        }

        Trajectory::Trajectory(const int &traj_type,
                               const double &traj_duration) : trajType(traj_type),
                                                              trajDuration(traj_duration)
        {
        }

        Trajectory::~Trajectory()
        {
        }

        nav_msgs::Path Trajectory::toRosPath() const
        {
            nav_msgs::Path path_msg;
            ros::Time t = ros::Time::now();
            path_msg.header.stamp = t;
            path_msg.header.frame_id = "world";

            geometry_msgs::PoseStamped pose;
            for (const auto &point : points)
            {
                pose.pose.position.x = point.position.x();
                pose.pose.position.y = point.position.y();
                pose.pose.position.z = point.position.z();
                pose.pose.orientation.w = point.orientation.w();
                pose.pose.orientation.x = point.orientation.x();
                pose.pose.orientation.y = point.orientation.y();
                pose.pose.orientation.z = point.orientation.z();
                path_msg.poses.push_back(pose);
            }
            return path_msg;
        }

    } // namespace common
} // namespace trajectory