#pragma once

#include <ros/ros.h>
#include <chrono>
#include <thread>

#include <minco/minco.hpp>
#include <minco/lbfgs.hpp>

using namespace std;

namespace minco_utils
{
  class TrajOpt
  {
  public:
    int N, K, dim_t;
    bool pause_debug;

    // Weight for time regularization term
    double rhoT;

    // Dynamics paramters
    double vmax, amax;
    double rhoV, rhoA;

    // Minimum Jerk Optimizer
    minco_utils::MinJerkOpt jerkOpt;

    // col(0) is P of (x,y,z), col(1) is V .. ， col(2) is A
    Eigen::MatrixXd initS;
    Eigen::MatrixXd finalS;

    // Duration of each piece of the trajectory
    double *x;

    Eigen::MatrixXd inPs;

  public:
    TrajOpt() {}
    ~TrajOpt() {}

    void setParam(const int &K,
                  const bool &pause_debug,
                  const Eigen::VectorXd &data);

    bool generate_traj(const Eigen::MatrixXd &initState,
                       const Eigen::MatrixXd &finalState,
                       const std::vector<Eigen::Vector3d> &Q,
                       const std::vector<double> &allocateT,
                       const int N,
                       minco_utils::Trajectory &traj,
                       bool keep_result,
                       int &ret_value);

    void addTimeIntPenalty(double &cost);

    bool grad_cost_v(const Eigen::Vector3d &v,
                     Eigen::Vector3d &gradv,
                     double &costv);

    bool grad_cost_a(const Eigen::Vector3d &a,
                     Eigen::Vector3d &grada,
                     double &costa);

  public:
    typedef shared_ptr<TrajOpt> Ptr;
  };
} // namespace minco_utils