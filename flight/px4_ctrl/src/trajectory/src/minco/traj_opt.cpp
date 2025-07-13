#include <minco/traj_opt.h>

using namespace Eigen;

namespace minco_utils
{
void TrajOpt::setParam(const int &K, 
                       const bool &pause_debug,
                       const Eigen::VectorXd &data)
{
  this->K = K;
  this->vmax = data(0);
  this->amax = data(1);
  this->rhoT = data(2);
  this->rhoV = data(3);
  this->rhoA = data(4);
  this->pause_debug = pause_debug;
}

void TrajOpt::addTimeIntPenalty(double &cost)
{
  double omg;
  double cost_tmp;
  double step, alpha;
  double s1, s2, s3, s4, s5;
  double gradViolaVt, gradViolaAt;
  Eigen::Vector3d grad_tmp;
  Eigen::Vector3d pos, vel, acc, jer;
  Eigen::Matrix<double, 6, 3> gradViolaVc, gradViolaAc;
  Eigen::Matrix<double, 6, 1> beta0, beta1, beta2, beta3;

  int innerLoop;
  for (int i = 0; i < N; ++i)
  {
    const auto &c = jerkOpt.b.block<6, 3>(i * 6, 0);
    step = jerkOpt.T1(i) / K;
    s1 = 0.0;
    innerLoop = K + 1;

    for (int j = 0; j < innerLoop; ++j)
    {
      s2 = s1 * s1;
      s3 = s2 * s1;
      s4 = s2 * s2;
      s5 = s4 * s1;
      beta0 << 1.0, s1, s2, s3, s4, s5;
      beta1 << 0.0, 1.0, 2.0 * s1, 3.0 * s2, 4.0 * s3, 5.0 * s4;
      beta2 << 0.0, 0.0, 2.0, 6.0 * s1, 12.0 * s2, 20.0 * s3;
      beta3 << 0.0, 0.0, 0.0, 6.0, 24.0 * s1, 60.0 * s2;
      alpha = 1.0 / K * j;
      pos = c.transpose() * beta0;
      vel = c.transpose() * beta1;
      acc = c.transpose() * beta2;
      jer = c.transpose() * beta3;

      omg = (j == 0 || j == innerLoop - 1) ? 0.5 : 1.0;

      if (grad_cost_v(vel, grad_tmp, cost_tmp))
      {
        gradViolaVc = beta1 * grad_tmp.transpose();
        gradViolaVt = alpha * grad_tmp.dot(acc);
        (this->jerkOpt).gdC.block<6, 3>(i * 6, 0) += omg * step * gradViolaVc;
        (this->jerkOpt).gdT(i) += omg * (cost_tmp / K + step * gradViolaVt);
        cost += omg * step * cost_tmp;
      }

      if (grad_cost_a(acc, grad_tmp, cost_tmp))
      {
        gradViolaAc = beta2 * grad_tmp.transpose();
        gradViolaAt = alpha * grad_tmp.dot(jer);
        (this->jerkOpt).gdC.block<6, 3>(i * 6, 0) += omg * step * gradViolaAc;
        (this->jerkOpt).gdT(i) += omg * (cost_tmp / K + step * gradViolaAt);
        cost += omg * step * cost_tmp;
      }
      s1 += step;
    }
  }
}

// SECTION variables transformation and gradient transmission
static double expC2(double t)
{
  return t > 0.0 ? ((0.5 * t + 1.0) * t + 1.0) : 1.0 / ((0.5 * t - 1.0) * t + 1.0);
}

static double logC2(double T)
{
  return T > 1.0 ? (sqrt(2.0 * T - 1.0) - 1.0) : (1.0 - sqrt(2.0 / T - 1.0));
}

static inline double gdT2t(double t)
{
  if (t > 0)
  {
    return t + 1.0;
  }
  else
  {
    double denSqrt = (0.5 * t - 1.0) * t + 1.0;
    return (1.0 - t) / (denSqrt * denSqrt);
  }
}

static void forwardT(const Eigen::VectorXd &t, Eigen::Ref<Eigen::VectorXd> vecT)
{
  for (int i = 0; i < t.size(); i++)
    vecT(i) = expC2(t(i));
}

// SECTION object function
static inline double objectiveFunc(void *ptrObj,
                                   const double *x,
                                   double *grad,
                                   const int n)
{
  TrajOpt &obj = *(TrajOpt *)ptrObj;
  VectorXd VT(obj.N);
  Eigen::Map<const Eigen::MatrixXd> T(x, 1, (obj.dim_t));
  Eigen::Map<Eigen::MatrixXd> gradT(grad, 1, (obj.dim_t));

  Eigen::VectorXd t(obj.N);
  for (int i = 0; i < obj.N; i++)
    t(i) = T(i);
  forwardT(t, VT);

  (obj.jerkOpt).generate(obj.inPs, (obj.finalS), VT);

  double cost = (obj.jerkOpt).getTrajJerkCost();

  (obj.jerkOpt).calGrads_CT();

  obj.addTimeIntPenalty(cost);

  (obj.jerkOpt).calGrads_PT();
  (obj.jerkOpt).gdT.array() += (obj.rhoT);

  cost += (obj.rhoT) * VT.sum();

  for (int i = 0; i < t.size(); i++)
    gradT(i) = (obj.jerkOpt).gdT(i) * gdT2t(t(i));

  return cost;
}

bool TrajOpt::generate_traj(const Eigen::MatrixXd &initState,
                            const Eigen::MatrixXd &finalState,
                            const std::vector<Eigen::Vector3d> &Q,
                            const std::vector<double> &allocateT,
                            const int N,
                            minco_utils::Trajectory &traj,
                            bool keep_result,
                            int &ret_value)
{
  this->N = N;
  this->dim_t = N;
  this->x = new double[(this->dim_t)];

  VectorXd VT(N);
  Eigen::Map<Eigen::MatrixXd> T((this->x), 1, (this->dim_t));

  (this->initS) = initState;
  (this->finalS) = finalState;

  Eigen::VectorXd T0(N);
  for (int i = 0; i < N; i++)
    T0(i) = allocateT.at(i);

  for (int i = 0; i < N; i++)
    T(i) = logC2(T0(i));

  inPs.resize(3, N - 1);
  for (int i = 0; i < N - 1; i++)
    inPs.col(i) = Q[i];
  
  (this->jerkOpt).reset(initState, N);

  // NOTE optimization
  lbfgs::lbfgs_parameter_t lbfgs_params;
  lbfgs::lbfgs_load_default_parameters(&lbfgs_params);
  lbfgs_params.mem_size = 128;
  lbfgs_params.past = 3;
  lbfgs_params.g_epsilon = 1e-32;
  lbfgs_params.min_step = 1e-32;
  lbfgs_params.delta = 1e-5;
  lbfgs_params.line_search_type = 0;
  double minObjectiveXY;

  auto opt_ret1 = lbfgs::lbfgs_optimize((this->dim_t),
                                        this->x,
                                        &minObjectiveXY,
                                        &objectiveFunc, nullptr,
                                        nullptr, this, &lbfgs_params);

  // std::cout << "\033[32m"
  //           << "ret: " << opt_ret1 << "\033[0m" << std::endl;
  if (opt_ret1 == -1008)
  {
    std::cout << "generate trajectory failed! The front-end path may be impassable." << std::endl;
  }
  ret_value = opt_ret1;
  if (this->pause_debug)
  {
    std::this_thread::sleep_for(std::chrono::milliseconds(2000));
  }
  if (opt_ret1 < 0)
  {
    if (!keep_result)
    {
      delete[] this->x;
    }
    return false;
  }

  Eigen::VectorXd tempT(N);
  for (int i = 0; i < N; i++)
    tempT(i) = T(i);
  forwardT(tempT, VT);
  (this->jerkOpt).generate(inPs, finalState, VT);
  traj = (this->jerkOpt).getTraj();
  if (!keep_result)
  {
    delete[] this->x;
  }
  return true;
}

bool TrajOpt::grad_cost_v(const Eigen::Vector3d &v,
                          Eigen::Vector3d &gradv,
                          double &costv)
{
  double vpen = v.squaredNorm() - (this->vmax) * (this->vmax);

  if (vpen > 0)
  {
    gradv = this->rhoV * 6 * vpen * vpen * v;
    costv = this->rhoV * vpen * vpen * vpen;
    return true;
  }
  return false;
}

bool TrajOpt::grad_cost_a(const Eigen::Vector3d &a,
                          Eigen::Vector3d &grada,
                          double &costa)
{
  grada = Eigen::Vector3d::Zero();
  costa = 0;
  double apen = a.squaredNorm() - (this->amax) * (this->amax);

  if (apen > 0)
  {
    grada += (this->rhoA) * 6 * apen * apen * a;
    costa += (this->rhoA) * apen * apen * apen;
    return true;
  }
  return false;
}
} // namespace minco_utils