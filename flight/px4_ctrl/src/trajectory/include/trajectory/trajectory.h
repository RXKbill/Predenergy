/*
 * @Author: Xinwei Chen
 * @Date: 2024-07-07 16:48:40
 * @Last Modified by: Xinwei Chen
 * @Last Modified time: 2024-07-07 16:48:43
 */

#ifndef TRAJECTORY_H
#define TRAJECTORY_H

#include <minco/traj_gen.hpp>
#include <trajectory/common.h>
#include <quadrotor_msgs/PolynomialTrajectory.h>

namespace trajectory
{
    using PolyTrajectory = quadrotor_msgs::PolynomialTrajectory;
    using MincoTrajectory = minco_utils::Trajectory;

    using TrajectoryPoint_t = common::TrajectoryPoint;
    using Trajectory_t = common::Trajectory;

    /**
     * @brief Generate a minco original trajectory
     *
     * @param MincoTrajParams
     * @return MincoTrajectory
     */
    MincoTrajectory generateMincoOriTrajectory(const minco_utils::MincoTrajParams &data);

    /**
     * @brief Generate a polynomial trajectory based on minco trajectory generator
     *
     * @param MincoTrajParams
     * @return PolyTrajectory
     */
    PolyTrajectory generatePolyTrajectory(const minco_utils::MincoTrajParams &data);

    /**
     * @brief Generate a polynomial trajectory based on minco trajectory generator
     *
     * @param MincoTrajectory
     * @return PolyTrajectory
     */
    PolyTrajectory generatePolyTrajectory(const MincoTrajectory &traj);

    /**
     * @brief Generate a custom trajectory based on minco trajectory generator
     *
     * @param MincoTrajParams
     * @return Trajectory_t
     */
    Trajectory_t generateCustomTrajectory(const minco_utils::MincoTrajParams &data);

    /**
     * @brief Generate a trajectory based on minco trajectory generator
     *
     * @param MincoTrajParams
     * @param MincoTrajectory
     * @return Trajectory_t
     */
    Trajectory_t generateCustomTrajectory(const double &dt, const MincoTrajectory &traj);

    /**
     * @brief Get the current time trajectory point
     *
     * @param current_time
     * @return TrajectoryPoint_t
     */
    TrajectoryPoint_t getCurTimeTrajectoryPoint(const MincoTrajectory &traj, double current_time);

} // namespace trajectory

#endif // TRAJECTORY_H