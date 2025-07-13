/*
 * @Author: Xinwei Chen
 * @Date: 2024-07-07 16:48:40
 * @Last Modified by: Xinwei Chen
 * @Last Modified time: 2024-07-07 16:48:43
 */

#include "trajectory/trajectory.h"
#include <fstream>
#include <sstream>

namespace trajectory
{

  enum TrajType
  {
    CONTINUOUS = 0,
    DISCONTINUOUS
  };

  std::string trajTypeToString(TrajType type)
  {
    switch (type)
    {
    case CONTINUOUS:
      return "CONTINUOUS";
    case DISCONTINUOUS:
      return "DISCONTINUOUS";
    default:
      return "UNKNOWN";
    }
  }

  MincoTrajectory generateMincoOriTrajectory(const minco_utils::MincoTrajParams &data)
  {
    minco_utils::TrajGenerator traj_gen(data);
    MincoTrajectory traj = traj_gen.get_minco_traj();
    return traj;
  }

  PolyTrajectory generatePolyTrajectory(const minco_utils::MincoTrajParams &data)
  {
    minco_utils::TrajGenerator traj_gen(data);
    MincoTrajectory traj = traj_gen.get_minco_traj();

    Eigen::Isometry3d tfR2L = Eigen::Isometry3d::Identity();

    PolyTrajectory trajMsg;
    trajMsg.header.stamp = ros::Time::now();
    trajMsg.header.frame_id = "world";
    trajMsg.trajectory_id = 1;
    trajMsg.action = quadrotor_msgs::PolynomialTrajectory::ACTION_ADD;
    trajMsg.num_order = traj[0].getOrder();
    trajMsg.num_segment = traj.getPieceNum();
    Eigen::Vector3d initialVel, finalVel;
    initialVel = tfR2L * traj.getVel(0.0);
    finalVel = tfR2L * traj.getVel(traj.getTotalDuration());
    trajMsg.start_yaw = 0.0;
    trajMsg.final_yaw = 0.0;

    for (size_t p = 0; p < (size_t)traj.getPieceNum(); p++)
    {
      trajMsg.time.push_back(traj[p].getDuration());
      trajMsg.order.push_back(traj[p].getCoeffMat().cols() - 1);

      Eigen::VectorXd linearTr(2);
      linearTr << 0.0, trajMsg.time[p];
      std::vector<Eigen::VectorXd> linearTrCoeffs;
      linearTrCoeffs.emplace_back(1);
      linearTrCoeffs[0] << 1;
      for (size_t k = 0; k < trajMsg.order[p]; k++)
      {
        linearTrCoeffs.push_back(RootFinder::polyConv(linearTrCoeffs[k], linearTr));
      }

      Eigen::MatrixXd coefMat(3, traj[p].getCoeffMat().cols());
      for (int i = 0; i < coefMat.cols(); i++)
      {
        coefMat.col(i) = tfR2L.rotation() * traj[p].getCoeffMat().col(coefMat.cols() - i - 1).head<3>();
      }
      coefMat.col(0) = (coefMat.col(0) + tfR2L.translation()).eval();

      for (int i = 0; i < coefMat.cols(); i++)
      {
        double coefx(0.0), coefy(0.0), coefz(0.0);
        for (int j = i; j < coefMat.cols(); j++)
        {
          coefx += coefMat(0, j) * linearTrCoeffs[j](i);
          coefy += coefMat(1, j) * linearTrCoeffs[j](i);
          coefz += coefMat(2, j) * linearTrCoeffs[j](i);
        }
        trajMsg.coef_x.push_back(coefx);
        trajMsg.coef_y.push_back(coefy);
        trajMsg.coef_z.push_back(coefz);
      }
    }

    trajMsg.mag_coeff = 1.0;
    trajMsg.debug_info = "";

    return trajMsg;
  }

  PolyTrajectory generatePolyTrajectory(const MincoTrajectory &traj)
  {
    Eigen::Isometry3d tfR2L = Eigen::Isometry3d::Identity();

    PolyTrajectory trajMsg;
    trajMsg.header.stamp = ros::Time::now();
    trajMsg.header.frame_id = "world";
    trajMsg.trajectory_id = 1;
    trajMsg.action = quadrotor_msgs::PolynomialTrajectory::ACTION_ADD;
    trajMsg.num_order = traj[0].getOrder();
    trajMsg.num_segment = traj.getPieceNum();
    Eigen::Vector3d initialVel, finalVel;
    initialVel = tfR2L * traj.getVel(0.0);
    finalVel = tfR2L * traj.getVel(traj.getTotalDuration());
    trajMsg.start_yaw = 0.0;
    trajMsg.final_yaw = 0.0;

    for (size_t p = 0; p < (size_t)traj.getPieceNum(); p++)
    {
      trajMsg.time.push_back(traj[p].getDuration());
      trajMsg.order.push_back(traj[p].getCoeffMat().cols() - 1);

      Eigen::VectorXd linearTr(2);
      linearTr << 0.0, trajMsg.time[p];
      std::vector<Eigen::VectorXd> linearTrCoeffs;
      linearTrCoeffs.emplace_back(1);
      linearTrCoeffs[0] << 1;
      for (size_t k = 0; k < trajMsg.order[p]; k++)
      {
        linearTrCoeffs.push_back(RootFinder::polyConv(linearTrCoeffs[k], linearTr));
      }

      Eigen::MatrixXd coefMat(3, traj[p].getCoeffMat().cols());
      for (int i = 0; i < coefMat.cols(); i++)
      {
        coefMat.col(i) = tfR2L.rotation() * traj[p].getCoeffMat().col(coefMat.cols() - i - 1).head<3>();
      }
      coefMat.col(0) = (coefMat.col(0) + tfR2L.translation()).eval();

      for (int i = 0; i < coefMat.cols(); i++)
      {
        double coefx(0.0), coefy(0.0), coefz(0.0);
        for (int j = i; j < coefMat.cols(); j++)
        {
          coefx += coefMat(0, j) * linearTrCoeffs[j](i);
          coefy += coefMat(1, j) * linearTrCoeffs[j](i);
          coefz += coefMat(2, j) * linearTrCoeffs[j](i);
        }
        trajMsg.coef_x.push_back(coefx);
        trajMsg.coef_y.push_back(coefy);
        trajMsg.coef_z.push_back(coefz);
      }
    }

    trajMsg.mag_coeff = 1.0;
    trajMsg.debug_info = "";

    return trajMsg;
  }

  Trajectory_t generateCustomTrajectory(const minco_utils::MincoTrajParams &data)
  {
    minco_utils::TrajGenerator traj_gen(data);
    MincoTrajectory traj = traj_gen.get_minco_traj();

    int traj_type = traj.getTrajType();
    double traj_duration = traj.getTotalDuration();
    Trajectory_t trajMsg(traj_type, traj_duration);

    // Check if the trajectory is correct
    if (traj_duration < data.dt)
    {
      ROS_ERROR("Trajectory duration is less than dt!");
      exit(1);
    }
    else
    {
      ROS_WARN("Trajectory type: %s, duration: %f",
               trajTypeToString((TrajType)traj_type).c_str(),
               traj_duration);
    }

    for (double t = 0.0; t < traj_duration; t += data.dt)
    {
      double temp = t;
      TrajectoryPoint_t point;
      if (temp <= traj_duration)
      {
        Eigen::Vector3d pr = traj.getPos(temp - data.dt);
        Eigen::Vector3d po = traj.getPos(temp);
        Eigen::Vector3d vo = traj.getVel(temp);
        Eigen::Vector3d ao = traj.getAcc(temp);

        point.position(0) = po(0);
        point.position(1) = po(1);
        point.position(2) = po(2);
        point.velocity(0) = vo(0);
        point.velocity(1) = vo(1);
        point.velocity(2) = vo(2);
        point.acceleration(0) = ao(0);
        point.acceleration(1) = ao(1);
        point.acceleration(2) = ao(2);

        double euler_yaw = atan2(po[1] - pr[1], po[0] - pr[0]);
        double G = 9.8;
        Eigen::Matrix3d W_R_B;
        Eigen::Vector3d tempAcc, x_B, y_B, z_B, x_C;
        tempAcc << ao(0), ao(1), ao(2) + G;
        z_B = tempAcc.normalized();
        x_C << cos(euler_yaw), sin(euler_yaw), 0.0;
        y_B = z_B.cross(x_C).normalized();
        x_B = y_B.cross(z_B).normalized();
        W_R_B << x_B, y_B, z_B;
        point.orientation = W_R_B;
        point.orientation.normalize();
      }
      else
      {
        Eigen::Vector3d pr = traj.getPos(traj_duration - data.dt);
        Eigen::Vector3d po = traj.getPos(traj_duration);
        Eigen::Vector3d vo = traj.getVel(traj_duration);
        Eigen::Vector3d ao = traj.getAcc(traj_duration);

        point.position(0) = po(0);
        point.position(1) = po(1);
        point.position(2) = po(2);
        point.velocity(0) = vo(0);
        point.velocity(1) = vo(1);
        point.velocity(2) = vo(2);
        point.acceleration(0) = ao(0);
        point.acceleration(1) = ao(1);
        point.acceleration(2) = ao(2);

        double euler_yaw = atan2(po[1] - pr[1], po[0] - pr[0]);
        double G = 9.8;
        Eigen::Matrix3d W_R_B;
        Eigen::Vector3d tempAcc, x_B, y_B, z_B, x_C;
        tempAcc << ao(0), ao(1), ao(2) + G;
        z_B = tempAcc.normalized();
        x_C << cos(euler_yaw), sin(euler_yaw), 0.0;
        y_B = z_B.cross(x_C).normalized();
        x_B = y_B.cross(z_B).normalized();
        W_R_B << x_B, y_B, z_B;
        point.orientation = W_R_B;
        point.orientation.normalize();
      }
      trajMsg.points.push_back(point);
    }
    return trajMsg;
  }

  Trajectory_t generateCustomTrajectory(const double &dt, const MincoTrajectory &traj)
  {
    int traj_type = traj.getTrajType();
    double traj_duration = traj.getTotalDuration();
    Trajectory_t trajMsg(traj_type, traj_duration);

    // Check if the trajectory is correct
    if (traj_duration < dt)
    {
      ROS_ERROR("Trajectory duration is less than dt!");
      exit(1);
    }
    else
    {
      ROS_WARN("Trajectory type: %s, duration: %f",
               trajTypeToString((TrajType)traj_type).c_str(),
               traj_duration);
    }

    for (double t = 0.0; t < traj_duration; t += dt)
    {
      double temp = t;
      TrajectoryPoint_t point;
      if (temp <= traj_duration)
      {
        Eigen::Vector3d pr = traj.getPos(temp - dt);
        Eigen::Vector3d po = traj.getPos(temp);
        Eigen::Vector3d vo = traj.getVel(temp);
        Eigen::Vector3d ao = traj.getAcc(temp);

        point.position(0) = po(0);
        point.position(1) = po(1);
        point.position(2) = po(2);
        point.velocity(0) = vo(0);
        point.velocity(1) = vo(1);
        point.velocity(2) = vo(2);
        point.acceleration(0) = ao(0);
        point.acceleration(1) = ao(1);
        point.acceleration(2) = ao(2);

        double euler_yaw = atan2(po[1] - pr[1], po[0] - pr[0]);
        double G = 9.8;
        Eigen::Matrix3d W_R_B;
        Eigen::Vector3d tempAcc, x_B, y_B, z_B, x_C;
        tempAcc << ao(0), ao(1), ao(2) + G;
        z_B = tempAcc.normalized();
        x_C << cos(euler_yaw), sin(euler_yaw), 0.0;
        y_B = z_B.cross(x_C).normalized();
        x_B = y_B.cross(z_B).normalized();
        W_R_B << x_B, y_B, z_B;
        point.orientation = W_R_B;
        point.orientation.normalize();
      }
      else
      {
        Eigen::Vector3d pr = traj.getPos(traj_duration - dt);
        Eigen::Vector3d po = traj.getPos(traj_duration);
        Eigen::Vector3d vo = traj.getVel(traj_duration);
        Eigen::Vector3d ao = traj.getAcc(traj_duration);

        point.position(0) = po(0);
        point.position(1) = po(1);
        point.position(2) = po(2);
        point.velocity(0) = vo(0);
        point.velocity(1) = vo(1);
        point.velocity(2) = vo(2);
        point.acceleration(0) = ao(0);
        point.acceleration(1) = ao(1);
        point.acceleration(2) = ao(2);

        double euler_yaw = atan2(po[1] - pr[1], po[0] - pr[0]);
        double G = 9.8;
        Eigen::Matrix3d W_R_B;
        Eigen::Vector3d tempAcc, x_B, y_B, z_B, x_C;
        tempAcc << ao(0), ao(1), ao(2) + G;
        z_B = tempAcc.normalized();
        x_C << cos(euler_yaw), sin(euler_yaw), 0.0;
        y_B = z_B.cross(x_C).normalized();
        x_B = y_B.cross(z_B).normalized();
        W_R_B << x_B, y_B, z_B;
        point.orientation = W_R_B;
        point.orientation.normalize();
      }
      trajMsg.points.push_back(point);
    }
    return trajMsg;
  }

  TrajectoryPoint_t getCurTimeTrajectoryPoint(const MincoTrajectory &traj, double current_time)

  {
    double traj_duration = traj.getTotalDuration();

    TrajectoryPoint_t point;
    if (current_time <= traj_duration)
    {
      Eigen::Vector3d po = traj.getPos(current_time);
      Eigen::Vector3d vo = traj.getVel(current_time);
      Eigen::Vector3d ao = traj.getAcc(current_time);

      point.position(0) = po(0);
      point.position(1) = po(1);
      point.position(2) = po(2);
      point.velocity(0) = vo(0);
      point.velocity(1) = vo(1);
      point.velocity(2) = vo(2);
      point.acceleration(0) = ao(0);
      point.acceleration(1) = ao(1);
      point.acceleration(2) = ao(2);
      point.orientation = Eigen::Quaterniond::Identity();
    }
    else
    {
      Eigen::Vector3d po = traj.getPos(traj_duration);
      Eigen::Vector3d vo = traj.getVel(traj_duration);
      Eigen::Vector3d ao = traj.getAcc(traj_duration);

      point.position(0) = po(0);
      point.position(1) = po(1);
      point.position(2) = po(2);
      point.velocity(0) = vo(0);
      point.velocity(1) = vo(1);
      point.velocity(2) = vo(2);
      point.acceleration(0) = ao(0);
      point.acceleration(1) = ao(1);
      point.acceleration(2) = ao(2);
      point.orientation = Eigen::Quaterniond::Identity();
    }
    return point;
  }
} // namespace trajectory