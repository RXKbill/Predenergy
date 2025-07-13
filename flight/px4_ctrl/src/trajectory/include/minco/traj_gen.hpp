/*
 * @Author: Xinwei Chen
 * @Date: 2024-02-19 14:24:22
 * @Last Modified by: Xinwei Chen
 * @Last Modified time: 2024-02-19 14:24:25
 */

#ifndef TRAJ_GEN_H
#define TRAJ_GEN_H

#include <minco/traj_opt.h>
#include <minco/minco.hpp>

#include <ros/ros.h>

namespace minco_utils
{

    struct MincoTrajParams
    {
        int K = 32;
        int shape = 0;
        double dt = 0.05;
        double rhoT = 1000;
        double rhoV = 20000;
        double rhoA = 20000;
        double maxVel = 1.0;
        double maxAcc = 1.0;
        double eightWidth = 3.0;
        double eightLength = 6.0;
        double circleRadius = 3.0;
        double starRadius = 3.0;
        double sWidth = 3.0;
        double sLength = 14.0;
        double startX = 0.0;
        double startY = 0.0;
        double endX = 0.0;
        double endY = 0.0;
        double height = 0.0;
        bool pauseDebug = false;

        static MincoTrajParams getMincoTrajParams(const ros::NodeHandle &pnh)
        {
            bool is_param_ok = true;
            int K = 0;
            int shape = 0;
            double dt = 0.0;
            double rhoT = 0.0;
            double rhoV = 0.0;
            double rhoA = 0.0;
            double maxVel = 0.0;
            double maxAcc = 0.0;
            double eightWidth = 0.0;
            double eightLength = 0.0;
            double circleRadius = 0.0;
            double starRadius = 0.0;
            double sWidth = 0.0;
            double sLength = 0.0;
            double startX = 0.0;
            double startY = 0.0;
            double endX = 0.0;
            double endY = 0.0;
            double height = 0.0;
            bool pauseDebug = false;

            is_param_ok &= pnh.getParam("K", K);
            is_param_ok &= pnh.getParam("shape", shape);
            is_param_ok &= pnh.getParam("dt", dt);
            is_param_ok &= pnh.getParam("rhoT", rhoT);
            is_param_ok &= pnh.getParam("rhoV", rhoV);
            is_param_ok &= pnh.getParam("rhoA", rhoA);
            is_param_ok &= pnh.getParam("maxVel", maxVel);
            is_param_ok &= pnh.getParam("maxAcc", maxAcc);
            is_param_ok &= pnh.getParam("eightWidth", eightWidth);
            is_param_ok &= pnh.getParam("eightLength", eightLength);
            is_param_ok &= pnh.getParam("circleRadius", circleRadius);
            is_param_ok &= pnh.getParam("starRadius", starRadius);
            is_param_ok &= pnh.getParam("sWidth", sWidth);
            is_param_ok &= pnh.getParam("sLength", sLength);
            is_param_ok &= pnh.getParam("startX", startX);
            is_param_ok &= pnh.getParam("startY", startY);
            is_param_ok &= pnh.getParam("endX", endX);
            is_param_ok &= pnh.getParam("endY", endY);
            is_param_ok &= pnh.getParam("height", height);
            is_param_ok &= pnh.getParam("pauseDebug", pauseDebug);

            if (!is_param_ok || K <= 0 || shape < 0 || shape > 4 || dt <= 0 || rhoT <= 0 || rhoV <= 0 || rhoA <= 0 ||
                maxVel <= 0 || maxAcc <= 0 || eightWidth <= 0 || eightLength <= 0 ||
                circleRadius <= 0 || starRadius <= 0 || sWidth <= 0 || sLength <= 0)
            {
                ROS_WARN("[MincoTrajParams::getMincoTrajParams] parameter error");
                exit(1);
            }

            return MincoTrajParams{K, shape, dt, rhoT, rhoV, rhoA, maxVel, maxAcc, eightWidth, eightLength,
                                   circleRadius, starRadius, sWidth, sLength, startX, startY, endX, endY, height,
                                   pauseDebug};
        }
    };

    enum TrajShape
    {
        EIGHT = 0,
        CIRCLE,
        STAR,
        SCURVE
    };

    enum TrajType
    {
        CONTINUOUS = 0,
        DISCONTINUOUS
    };

    class TrajGenerator
    {
    public:
        // constructor
        TrajGenerator() = delete;
        TrajGenerator(const MincoTrajParams &data);
        TrajGenerator(const TrajGenerator &rhs) = delete;
        TrajGenerator &operator=(const TrajGenerator &rhs) = delete;
        TrajGenerator(TrajGenerator &&rhs) = delete;
        TrajGenerator &operator=(TrajGenerator &&rhs) = delete;
        virtual ~TrajGenerator() {}

        // getters
        Trajectory get_minco_traj() const { return minco_traj_; }

    private:
        MincoTrajParams data_;
        Trajectory minco_traj_;

        void generate_traj();

    }; // class TrajGenerator

    void TrajGenerator::generate_traj()
    {
        // Define trajectory state
        Eigen::MatrixXd initState = Eigen::MatrixXd::Zero(3, 3);
        Eigen::MatrixXd finalState = Eigen::MatrixXd::Zero(3, 3);
        Eigen::Vector3d initStateV = Eigen::Vector3d::Zero();
        Eigen::Vector3d finalStateV = Eigen::Vector3d::Zero();

        std::vector<Eigen::Vector3d> Q;
        std::vector<double> T;
        int N = 0;

        int trajType;

        if (data_.shape == EIGHT) // Eight
        {
            initState(0, 0) = initStateV(0) = data_.startX;
            initState(1, 0) = initStateV(1) = data_.startY;
            initState(2, 0) = initStateV(2) = data_.height;
            finalState(0, 0) = finalStateV(0) = data_.startX;
            finalState(1, 0) = finalStateV(1) = data_.startY;
            finalState(2, 0) = finalStateV(2) = data_.height;

            double eightWidth = data_.eightWidth;
            double eightLength = data_.eightLength;
            double offsetX = data_.startX;
            double offsetY = data_.startY;

            Eigen::VectorXd PosX(7), PosY(7);
            PosX << eightLength / 4, eightLength / 2, eightLength / 4, 0.0, -eightLength / 4, -eightLength / 2, -eightLength / 4;
            PosY << -eightWidth / 2, 0.0, eightWidth / 2, 0.0, -eightWidth / 2, 0.0, eightWidth / 2;

            N = PosX.size() + 1;
            for (int i = 0; i < N - 1; i++)
            {
                Eigen::Vector3d wp(PosX(i) + offsetX,
                                   PosY(i) + offsetY,
                                   data_.height);
                Q.push_back(wp);
            }

            for (int i = 0; i < N; i++)
            {
                double dis;
                if (i == 0)
                    dis = (initStateV - Q[i]).norm();
                else if (i == N - 1)
                    dis = (finalStateV - Q[i - 1]).norm();
                else
                    dis = (Q[i] - Q[i - 1]).norm();
                double tempT = dis / data_.maxVel;
                T.push_back(tempT);
            }

            trajType = CONTINUOUS;
        }
        else if (data_.shape == CIRCLE) // Circle
        {
            double R = data_.circleRadius;

            initState(0, 0) = initStateV(0) = data_.startX;
            initState(1, 0) = initStateV(1) = data_.startY;
            initState(2, 0) = initStateV(2) = data_.height;
            finalState(0, 0) = finalStateV(0) = data_.startX;
            finalState(1, 0) = finalStateV(1) = data_.startY;
            finalState(2, 0) = finalStateV(2) = data_.height;

            double offsetX = data_.startX - R;
            double offsetY = data_.startY;

            N = 100;
            for (int i = 1; i < N; i++)
            {
                Eigen::Vector3d wp( R * cos(i * 2 * M_PI / N) + offsetX,
                                   -R * sin(i * 2 * M_PI / N) + offsetY,
                                    data_.height);
                Q.push_back(wp);
            }

            for (int i = 0; i < N; i++)
            {
                double dis;
                if (i == 0)
                    dis = (initStateV - Q[i]).norm();
                else if (i == N - 1)
                    dis = (finalStateV - Q[i - 1]).norm();
                else
                    dis = (Q[i] - Q[i - 1]).norm();
                double tempT = dis / data_.maxVel;
                T.push_back(tempT);
            }

            trajType = CONTINUOUS;
        }
        else if (data_.shape == STAR)
        {
            double R = data_.starRadius;
            double r = R * 0.65;

            initState(0, 0) = initStateV(0) = data_.startX;
            initState(1, 0) = initStateV(1) = data_.startY;
            initState(2, 0) = initStateV(2) = data_.height;
            finalState(0, 0) = finalStateV(0) = data_.startX;
            finalState(1, 0) = finalStateV(1) = data_.startY;
            finalState(2, 0) = finalStateV(2) = data_.height;

            double offsetX = data_.startX - R;
            double offsetY = data_.startY;

            Eigen::VectorXd PosX(9), PosY(9);
            PosX << r * sin(54 * M_PI / 180), R * sin(18 * M_PI / 180), -r * sin(18 * M_PI / 180), -R * sin(54 * M_PI / 180), -r, -R * sin(54 * M_PI / 180), -r * sin(18 * M_PI / 180), R * sin(18 * M_PI / 180), r * sin(54 * M_PI / 180);
            PosY << -r * cos(54 * M_PI / 180), -R * cos(18 * M_PI / 180), -r * cos(18 * M_PI / 180), -R * cos(54 * M_PI / 180), 0, R * cos(54 * M_PI / 180), r * cos(18 * M_PI / 180), R * cos(18 * M_PI / 180), r * cos(54 * M_PI / 180);

            N = PosX.size() + 1;
            for (int i = 0; i < N - 1; i++)
            {
                Eigen::Vector3d wp(PosX(i) + offsetX,
                                   PosY(i) + offsetY,
                                   data_.height);
                Q.push_back(wp);
            }

            for (int i = 0; i < N; i++)
            {
                double dis;
                if (i == 0)
                    dis = (initStateV - Q[i]).norm();
                else if (i == N - 1)
                    dis = (finalStateV - Q[i - 1]).norm();
                else
                    dis = (Q[i] - Q[i - 1]).norm();
                double tempT = dis / data_.maxVel;
                T.push_back(tempT);
            }

            trajType = CONTINUOUS;
        }
        else if (data_.shape == SCURVE) // S-Curve
        {
            initState(0, 0) = initStateV(0) = data_.startX;
            initState(1, 0) = initStateV(1) = data_.startY;
            initState(2, 0) = initStateV(2) = data_.height;
            finalState(0, 0) = finalStateV(0) = 0.0; // modified later (finalState(0, 0) = sLength + offsetX;)
            finalState(1, 0) = finalStateV(1) = 0.0; // modified later (finalState(1, 0) = -sWidth + offsetY;)
            finalState(2, 0) = finalStateV(2) = data_.height;

            double sWidth = data_.sWidth;
            double sLength = data_.sLength;
            double sPerLength = sLength / 7;
            double offsetX = data_.startX;
            double offsetY = data_.startY;

            finalState(0, 0) = sLength + offsetX;
            finalState(1, 0) = -sWidth + offsetY;

            Eigen::VectorXd PosX(6), PosY(6);
            PosX << sPerLength, 2 * sPerLength, 3 * sPerLength, 4 * sPerLength, 5 * sPerLength, 6 * sPerLength;
            PosY << -sWidth, -sWidth / 2, 0.0, -sWidth, -sWidth / 2, 0.0;

            N = PosX.size() + 1;
            for (int i = 0; i < N - 1; i++)
            {
                Eigen::Vector3d wp(PosX(i) + offsetX,
                                   PosY(i) + offsetY,
                                   data_.height);
                Q.push_back(wp);
            }

            for (int i = 0; i < N; i++)
            {
                double dis;
                if (i == 0)
                    dis = (initStateV - Q[i]).norm();
                else if (i == N - 1)
                    dis = (finalStateV - Q[i - 1]).norm();
                else
                    dis = (Q[i] - Q[i - 1]).norm();
                double tempT = dis / data_.maxVel;
                T.push_back(tempT);
            }

            trajType = DISCONTINUOUS;
        }
        else
        {
            ROS_WARN("Don't exist this trajectory type!!!");
        }

        // Set trajectory
        TrajOpt::Ptr minco_traj_optimizer;

        minco_traj_optimizer.reset(new TrajOpt);
        int ret_value;
        bool ret_opt;

        Eigen::VectorXd minco_param(5);
        minco_param << data_.maxVel, data_.maxAcc, data_.rhoT, data_.rhoV, data_.rhoA;
        minco_traj_optimizer->setParam(data_.K, data_.pauseDebug, minco_param);

        ret_opt = minco_traj_optimizer->generate_traj(initState, finalState, Q, T, N, minco_traj_, false, ret_value);
        minco_traj_.setTrajType(trajType);
    }

    TrajGenerator::TrajGenerator(const MincoTrajParams &data) : data_(data)
    {
        generate_traj();
    }

} // namespace minco_utils

#endif // TRAJ_GEN_H
