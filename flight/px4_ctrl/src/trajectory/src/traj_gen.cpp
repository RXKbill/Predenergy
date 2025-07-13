/*
 * @Brief:
 * @Author: Xinwei Chen
 * @Date: 2024-07-07 17:34:16
 * @Last Modified by: Xinwei Chen
 * @Last Modified time: 2024-07-08 15:49:31
 */

#include <ros/ros.h>

#include <tf/tf.h>
#include <std_msgs/Bool.h>
#include <geometry_msgs/PoseStamped.h>

#include <minco/traj_gen.hpp>
#include <trajectory/common.h>
#include <trajectory/visual.h>
#include <trajectory/trajectory.h>

#include <Eigen/Eigen>

int main(int argc, char **argv)
{
    ros::init(argc, argv, "pubtraj");

    ros::NodeHandle nh, pnh("~");
    ros::AsyncSpinner spinner(1);

    // publush the trajectory
    ros::Publisher traj_pub = nh.advertise<quadrotor_msgs::PolynomialTrajectory>("trajectory", 1);
    ros::Publisher cmd_vis_pub = nh.advertise<visualization_msgs::Marker>("cmd_visual", 1);

    minco_utils::MincoTrajParams poly_config_data = std::move(minco_utils::MincoTrajParams::getMincoTrajParams(pnh));
    minco_utils::Trajectory FOLLOW_TRAJ_MINCO = trajectory::generateMincoOriTrajectory(poly_config_data);

    quadrotor_msgs::PolynomialTrajectory FOLLOW_TRAJ_POLY = trajectory::generatePolyTrajectory(FOLLOW_TRAJ_MINCO);
    trajectory::common::Trajectory FOLLOW_TRAJ_CUSTOM = trajectory::generateCustomTrajectory(poly_config_data.dt, FOLLOW_TRAJ_MINCO);

    trajectory::visual::TrajectoryVisualizer traj_visualizer("trajectory_vis", nh);
    traj_visualizer.setTrajectory(FOLLOW_TRAJ_CUSTOM);

    auto START_TIME = ros::Time::now();

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
    ros::Rate rate(100);

    while (ros::ok())
    {
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