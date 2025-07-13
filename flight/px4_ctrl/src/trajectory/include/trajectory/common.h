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

#pragma once

#include <vector>
#include <Eigen/Eigen>

#include <ros/ros.h>
#include <nav_msgs/Path.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/Twist.h>

namespace trajectory
{
    namespace common
    {
        /**
         * @brief Trajectory point
         *
         */
        struct TrajectoryPoint
        {
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW

            TrajectoryPoint();
            TrajectoryPoint(const Eigen::Vector3d &p,
                            const Eigen::Vector3d &v,
                            const Eigen::Vector3d &a,
                            const Eigen::Vector3d &j,
                            const Eigen::Quaterniond &q);
            virtual ~TrajectoryPoint();

            Eigen::Vector3d position;
            Eigen::Vector3d velocity;
            Eigen::Vector3d acceleration;
            Eigen::Vector3d jerk;
            Eigen::Quaterniond orientation;
        };

        /**
         * @brief Trajectory
         *
         */
        struct Trajectory
        {
            Trajectory();
            Trajectory(const trajectory::common::TrajectoryPoint &point);
            Trajectory(const int &traj_type,
                       const double &traj_duration);
            virtual ~Trajectory();

            nav_msgs::Path toRosPath() const;

            std::vector<trajectory::common::TrajectoryPoint> points;
            int trajType;
            double trajDuration;
        };

    } // namespace common
} // namespace trajectory