/*
 * @Author: Xinwei Chen
 * @Date: 2024-07-07 16:30:42
 * @Last Modified by: Xinwei Chen
 * @Last Modified time: 2024-07-07 16:30:38
 */

#ifndef TRAJECTORY_SIMULATION_H
#define TRAJECTORY_SIMULATION_H

#include <string>

#include <Eigen/Eigen>

#include <nav_msgs/Odometry.h>
#include <gazebo_msgs/ModelStates.h>

namespace trajectory
{
    namespace sim
    {

        /**
         * @brief Generate Odometry from Gazebo ModelStates message.
         * The velocity generated from this function is in the **WORLD** frame.
         *
         * @param gz_msg_ptr Gazebo ModelStates message pointer
         * @param model_name The name of the model
         * @return nav_msgs::Odometry
         */
        nav_msgs::Odometry genOdomFromGZ(const gazebo_msgs::ModelStates::ConstPtr &gz_msg_ptr,
                                         const std::string &model_name);

    } // namespace sim
} // namespace trajectory

#endif // TRAJECTORY_SIMULATION_H