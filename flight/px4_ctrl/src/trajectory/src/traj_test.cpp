/*
 * @Brief:
 * @Author: Xinwei Chen
 * @Date: 2024-02-19 15:57:08
 * @Last Modified by: Xinwei Chen
 * @Last Modified time: 2024-07-08 16:38:28
 */

#include "trajectory/common.h"
#include "trajectory/visual.h"
#include "trajectory/trajectory.h"
#include "trajectory/simulation.h"

#include <Eigen/Eigen>
#include <minco/traj_gen.hpp>

#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <gazebo_msgs/ModelStates.h>
#include <visualization_msgs/Marker.h>

int main(int argc, char **argv)
{
    ros::init(argc, argv, "traj_test_node");

    ros::NodeHandle nh, pnh("~");
    ros::AsyncSpinner spinner(1);

    // publush the trajectory
    ros::Publisher traj_pub = nh.advertise<quadrotor_msgs::PolynomialTrajectory>("trajectory", 1);
    ros::Publisher cmd_vis_pub = nh.advertise<visualization_msgs::Marker>("cmd_visual", 1);
    ros::Publisher odom_pub = nh.advertise<nav_msgs::Odometry>("odometry", 1);

    minco_utils::MincoTrajParams poly_config_data = std::move(minco_utils::MincoTrajParams::getMincoTrajParams(pnh));
    minco_utils::Trajectory FOLLOW_TRAJ_MINCO = trajectory::generateMincoOriTrajectory(poly_config_data);

    quadrotor_msgs::PolynomialTrajectory FOLLOW_TRAJ_POLY = trajectory::generatePolyTrajectory(FOLLOW_TRAJ_MINCO);
    trajectory::common::Trajectory FOLLOW_TRAJ_CUSTOM = trajectory::generateCustomTrajectory(poly_config_data.dt, FOLLOW_TRAJ_MINCO);

    trajectory::visual::TrajectoryVisualizer traj_visualizer("trajectory_vis", nh);
    traj_visualizer.setTrajectory(FOLLOW_TRAJ_CUSTOM);

    auto START_TIME = ros::Time::now();

    // subscribe to the gazebo model states
    nav_msgs::Odometry QUAD_ODOM;
    const std::string QUAD_MODEL_NAME = "iris_livox";
    const std::string GAZEBO_MODELS_TOPIC = "/gazebo/model_states";
    ros::Subscriber gazebo_model_states_sub = nh.subscribe<gazebo_msgs::ModelStates>(
        GAZEBO_MODELS_TOPIC, 1,
        [&QUAD_ODOM, QUAD_MODEL_NAME](const gazebo_msgs::ModelStates::ConstPtr &gz_msg_ptr)
        {
            QUAD_ODOM = trajectory::sim::genOdomFromGZ(gz_msg_ptr, QUAD_MODEL_NAME);
        });
    while (!gazebo_model_states_sub.getNumPublishers())
    {
        ros::Rate(1).sleep();
        ROS_WARN("Waiting for connection");
    }

    // subscribe to the run trigger
    geometry_msgs::PoseStamped RUN_TRIGGER_MSG;
    const std::string RUN_TRIGGER_TOPIC = "move_base_simple/goal";
    ros::Subscriber run_trigger_sub = nh.subscribe<geometry_msgs::PoseStamped>(
        RUN_TRIGGER_TOPIC, 1,
        [&traj_pub, &FOLLOW_TRAJ_POLY, &traj_visualizer, &START_TIME, &RUN_TRIGGER_MSG](const geometry_msgs::PoseStamped::ConstPtr &msg_ptr)
        {
            ROS_WARN("Receive a new trigger, publishing the trajectory!");
            traj_pub.publish(FOLLOW_TRAJ_POLY);
            traj_visualizer.publishTrajectory();
            START_TIME = ros::Time::now();
            RUN_TRIGGER_MSG = *msg_ptr;
        });

    visualization_msgs::Marker cmd_vis_marker;

    spinner.start();
    ros::Rate rate(200);

    while (ros::ok())
    {
        rate.sleep();

        QUAD_ODOM.header.frame_id = "world";
        odom_pub.publish(QUAD_ODOM);

        if (RUN_TRIGGER_MSG.header.stamp.toSec() == 0.0)
        {
            START_TIME = ros::Time::now();
            continue;
        }

        auto current_time = (ros::Time::now() - START_TIME).toSec();
        auto cur_traj_point = trajectory::getCurTimeTrajectoryPoint(FOLLOW_TRAJ_MINCO, current_time);
        Eigen::Vector3d euler = cur_traj_point.orientation.toRotationMatrix().eulerAngles(2, 1, 0);
        double yaw = euler[0];
        if (cur_traj_point.velocity.norm() > 1.0)
            cmd_vis_marker = trajectory::visual::genVisualMarker(cur_traj_point.position,
                                                                 cur_traj_point.position + cur_traj_point.velocity.normalized().norm() * Eigen::Vector3d(cos(yaw), sin(yaw), 0.0),
                                                                 trajectory::visual::VISUAL_COLOR::RED,
                                                                 0);
        else
            cmd_vis_marker = trajectory::visual::genVisualMarker(cur_traj_point.position,
                                                                 cur_traj_point.position + cur_traj_point.velocity.norm() * Eigen::Vector3d(cos(yaw), sin(yaw), 0.0),
                                                                 trajectory::visual::VISUAL_COLOR::RED,
                                                                 0);

        cmd_vis_pub.publish(cmd_vis_marker);

        ros::spinOnce();
    }
    spinner.stop();
    return 0;
}
