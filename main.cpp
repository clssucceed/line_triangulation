#include <iostream>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <array>
#include <ctime>
#include <cmath>
#include <unistd.h>

constexpr double kImageWidth = 1280;
constexpr double kImageHeight = 720;
constexpr double kFocalLength = 800;
// 1pixel的噪声会产生5度以内的方向误差，对点到直线的距离的影响在1%以内
constexpr double noise = 0.0 / kFocalLength;
// Question: 单目时V倒数第二个列向量才是真正的解; 双目时V的倒数第一个列向量是真正的解
constexpr bool kUseStereo = true;
constexpr int kVIndex = 5;

Eigen::Matrix3d SkewMatrix(const Eigen::Vector3d& v) {
    Eigen::Matrix3d m;
    m << 0, -v(2), v(1), v(2), 0, -v(0), -v(1), v(0), 0;
    // std::cout << v.transpose() << std::endl;
    // std::cout << m << std::endl;
    return m;
}

double AngleBetweenTwoVectors(const Eigen::Vector3d& v1, const Eigen::Vector3d& v2) {
    double cos_theta = v1.normalized().dot(v2.normalized());
    double theta = std::fabs(std::acos(cos_theta) * 180 / M_PI);
    if (theta > 90) {
        theta = 180 - theta;
    }
    return theta;
}

void GenerateTwoPoints(const Eigen::Vector3d &ni, Eigen::Vector3d& si, Eigen::Vector3d& ei) {
    assert(std::fabs(ni(0)) > 1.0e-6 || std::fabs(ni(1)) > 1.0e-6);
    if (std::fabs(ni(0)) < 1.0e-6) {
        // 图像中的水平线
        si(1) = -ni(2) / ni(1);
        assert(std::fabs(si(1)) < kImageHeight * 0.5 / kFocalLength);
        ei(1) = si(1);
        si(0) = -kImageWidth * 0.25 / kFocalLength;
        ei(0) = -si(0);
        si(2) = 1;
        ei(2) = 1;
        std::cout << "ni: " << ni.transpose() << std::endl;
        std::cout << "si: " << si.transpose() << std::endl;
        double si_theta = ni.dot(si) / (ni.norm() * si.norm());
        std::cout << "si_theta:" << si_theta << std::endl;
        assert(si_theta < 1.0e-6);
        std::cout << "ei:" << ei.transpose() << std::endl;
        double ei_theta = ni.dot(ei) / (ni.norm() * ei.norm());
        std::cout << "ei_theta:" << ei_theta << std::endl;
        assert(ei_theta < 1.0e-6);
        return; 
    }
    if (std::fabs(ni(1)) < 1.0e-6) {
        // 图像中的竖直线
        si(0) = -ni(2) / ni(0);
        assert(std::fabs(si(0)) < kImageWidth * 0.5 / kFocalLength);
        ei(0) = si(0);
        si(1) = -kImageHeight * 0.25 / kFocalLength;
        ei(1) = -si(1);
        si(2) = 1;
        ei(2) = 1;
        std::cout << "ni: " << ni.transpose() << std::endl;
        std::cout << "si: " << si.transpose() << std::endl;
        double si_theta = ni.dot(si) / (ni.norm() * si.norm());
        std::cout << "si_theta:" << si_theta << std::endl;
        assert(si_theta < 1.0e-6);
        std::cout << "ei:" << ei.transpose() << std::endl;
        double ei_theta = ni.dot(ei) / (ni.norm() * ei.norm());
        std::cout << "ei_theta:" << ei_theta << std::endl;
        assert(ei_theta < 1.0e-6);
        return; 
    }
    double si_x = (-kImageWidth * 0.5 - 1) / kFocalLength, si_y = (-kImageHeight * 0.5 - 1) / kFocalLength;
    while (si_x < (-kImageWidth * 0.5) / kFocalLength || si_x > (kImageWidth * 0.5) / kFocalLength|| 
           si_y < (-kImageHeight * 0.5) / kFocalLength || si_y > (kImageHeight * 0.5) / kFocalLength) {
        si_x = (std::rand() % static_cast<int>(kImageWidth)) - kImageWidth * 0.5;
        si_x /= kFocalLength;
        si_y = -(ni(0) * si_x + ni(2)) / ni(1);
        std::cout << "si_x: " << si_x << ", si_y: " << si_y << std::endl;
        // sleep(1);
    }
    si = Eigen::Vector3d(si_x, si_y, 1);
    double ei_x = (-kImageHeight * 0.5 - 1) / kFocalLength, ei_y = (-kImageHeight * 0.5 - 1) / kFocalLength;
    while (ei_x * kFocalLength < -kImageHeight * 0.5 || ei_x * kFocalLength > kImageHeight * 0.5 || 
           ei_y * kFocalLength < -kImageHeight * 0.5 || ei_y * kFocalLength > kImageHeight * 0.5 ||
           std::fabs(si_x - ei_x) < 10 / kFocalLength) {
        ei_x = (std::rand() % static_cast<int>(kImageHeight)) - kImageHeight * 0.5;
        ei_x /= kFocalLength;
        ei_y = -(ni(0) * ei_x + ni(2)) / ni(1);
        std::cout << "ei_x: " << ei_x << ", ei_y: " << ei_y << std::endl;
        // sleep(1);
    }
    ei = Eigen::Vector3d(ei_x, ei_y, 1);
    std::cout << "ni: " << ni.transpose() << std::endl;
    std::cout << "si: " << si.transpose() << std::endl;
    double si_theta = ni.dot(si) / (ni.norm() * si.norm());
    std::cout << "si_theta:" << si_theta << std::endl;
    assert(si_theta < 1.0e-6);
    std::cout << "ei:" << ei.transpose() << std::endl;
    double ei_theta = ni.dot(ei) / (ni.norm() * ei.norm());
    std::cout << "ei_theta:" << ei_theta << std::endl;
    assert(ei_theta < 1.0e-6);
}

void GenerateLineObv(const Eigen::Matrix3d& Ri, const Eigen::Vector3d& ti, 
                     const Eigen::Vector3d& n, const Eigen::Vector3d& d, 
                     Eigen::Vector3d& si, Eigen::Vector3d& ei) {
     Eigen::Vector3d ni = Ri * n + SkewMatrix(ti) * Ri * d;
     GenerateTwoPoints(ni, si, ei);
     std::cout << "si: " << si.transpose() << std::endl;
     std::cout << "ei: " << ei.transpose() << std::endl;
     si.head(2) = si.head(2) + Eigen::Vector2d::Random().normalized() * noise;
     ei.head(2) = ei.head(2) + Eigen::Vector2d::Random().normalized() * noise;
     std::cout << "si_noised: " << si.transpose() << std::endl;
     std::cout << "ei_noised: " << ei.transpose() << std::endl;
}

int main(int, char**) {
    std::srand(std::time(nullptr));
    // step 1: 根据典型场景设置n,d
    const Eigen::Vector3d dx(1, 0, 0); // 混凝土交界处
    const Eigen::Vector3d dy(0, 1, 0); // 路灯杆
    const Eigen::Vector3d dz(0, 0, 1); // 车道线
    const Eigen::Vector3d nx(0, 10, -1.6); // nx和cx联合表示首帧相机系下前方10m的混凝土交界处
    const Eigen::Vector3d ny(10, 0, 5); // ny和cy联合表示首帧相机系下前方10m左侧5m的路灯杆
    const Eigen::Vector3d nz(1.6, 3, 0); // nz和cz联合表示首帧相机系下左侧3m的车道线
    Eigen::Vector3d d = dy;
    Eigen::Vector3d n = ny;
    // std::cout << n.transpose() << std::endl;
    // std::cout << d.transpose() << std::endl;
    // std::cout << n.transpose() * d << std::endl;
    // std::cout << n.norm() / d.norm() << std::endl;
    // std::cout << std::fabs(AngleBetweenTwoVectors(n, d)) << std::endl;
    assert(std::fabs(AngleBetweenTwoVectors(n, d) - 90) < 1.0e-6);

    // step 2: 根据高速典型运动设置Ri, ti并且同时生成si, ei 
    constexpr int kObvNum = 3;
    std::array<Eigen::Matrix3d, kObvNum> Rs;
    std::array<Eigen::Vector3d, kObvNum> ts;
    std::array<std::pair<Eigen::Vector3d, Eigen::Vector3d>, kObvNum> ses; // normalized
    std::array<Eigen::Matrix3d, kObvNum> Rs_r;
    std::array<Eigen::Vector3d, kObvNum> ts_r;
    std::array<std::pair<Eigen::Vector3d, Eigen::Vector3d>, kObvNum> ses_r;
    for (int i = 0; i < kObvNum; ++i) {
        // 设置Ri, ti
        // Twc
        std::cout << "##############################" << i << std::endl;
        Eigen::Matrix3d Ri = Eigen::Matrix3d::Identity();
        Eigen::Vector3d ti(0, 0, 1.0 * i); 
        Eigen::Matrix3d Ri_r = Ri;
        Eigen::Vector3d ti_r = ti + Eigen::Vector3d(0.2, 0, 0);
        // convert to Tcw
        Ri = Ri.transpose().eval();
        ti = -Ri * ti;
        Ri_r = Ri_r.transpose().eval();
        ti_r = -Ri_r * ti_r;
        std::cout << "Ri:" << std::endl << Ri << std::endl;
        std::cout << "ti:" << ti.transpose() << std::endl;
        std::cout << "Ri_r: " << std::endl << Ri_r << std::endl;
        std::cout << "ti_r: " << ti_r.transpose() << std::endl;
        // 生成si, ei
        Eigen::Vector3d si, ei;
        GenerateLineObv(Ri, ti, n, d, si, ei);
        Eigen::Vector3d si_r, ei_r;
        GenerateLineObv(Ri_r, ti_r, n, d, si_r, ei_r);
        // 保存Ri, ti, si, ei
        Rs[i] = Ri;
        ts[i] = ti;
        Rs_r[i] = Ri_r;
        ts_r[i] = ti_r;
        ses[i] = std::pair<Eigen::Vector3d, Eigen::Vector3d>(si, ei);
        ses_r[i] = std::pair<Eigen::Vector3d, Eigen::Vector3d>(si_r, ei_r);
    }

    // step 3: 生成线性求解系数矩阵,每一行是pi^T[Ri, [ti]_{\times}Ri][n, d]^T = 0;
    Eigen::MatrixXd A;
    if (kUseStereo) {
        Eigen::Matrix<double, 4 * kObvNum, 6> A_temp;
        for (int i = 0; i < kObvNum; ++i) {
            Eigen::Matrix<double, 3, 6> temp;
            temp.leftCols(3) = Rs.at(i);
            temp.rightCols(3) = SkewMatrix(ts.at(i)) * Rs.at(i); 
            A_temp.row(i * 4) = ses.at(i).first.transpose() * temp; 
            A_temp.row(i * 4 + 1) = ses.at(i).second.transpose() * temp; 
            Eigen::Matrix<double, 3, 6> temp_r;
            temp_r.leftCols(3) = Rs_r.at(i);
            temp_r.rightCols(3) = SkewMatrix(ts_r.at(i)) * Rs_r.at(i); 
            A_temp.row(i * 4 + 2) = ses_r.at(i).first.transpose() * temp_r; 
            A_temp.row(i * 4 + 3) = ses_r.at(i).second.transpose() * temp_r; 
        }
        A = A_temp;
    } else {
        Eigen::Matrix<double, 2 * kObvNum, 6> A_temp;
        for (int i = 0; i < kObvNum; ++i) {
            Eigen::Matrix<double, 3, 6> temp;
            temp.leftCols(3) = Rs.at(i);
            temp.rightCols(3) = SkewMatrix(ts.at(i)) * Rs.at(i); 
            A_temp.row(i * 2) = ses.at(i).first.transpose() * temp; 
            A_temp.row(i * 2 + 1) = ses.at(i).second.transpose() * temp; 
        }
        A = A_temp;
    }
    std::cout << "###########################################" << std::endl; 
    std::cout << "A:" << std::endl << A << std::endl;
    std::cout << "###########################################" << std::endl; 
    
    // step 4; 分析A的奇异值
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
    std::cout << "A.singular_values: " << svd.singularValues().transpose() << std::endl;
    std::cout << "svd.V: " << std::endl << svd.matrixV() << std::endl;
    Eigen::VectorXd L_est = svd.matrixV().col(kVIndex);
    std::cout << "L_est: " << L_est.normalized().transpose() << std::endl;
    Eigen::Matrix<double, 6, 1> L_gt;
    L_gt << n, d;
    std::cout << "L_gt: " << L_gt.normalized().transpose() << std::endl;
    Eigen::Vector3d n_est = L_est.head(3);
    Eigen::Vector3d d_est = L_est.tail(3);
    std::cout << "n_est_normalized: " << n_est.normalized().transpose() << std::endl;
    std::cout << "n_normalized: " << n.normalized().transpose() << std::endl;
    std::cout << "d_est_normalized: " << d_est.normalized().transpose() << std::endl;
    std::cout << "d_normalized: " << d.normalized().transpose() << std::endl;
    std::cout << "Angle_nd_est: " << AngleBetweenTwoVectors(n_est, d_est) << std::endl;
    std::cout << "Angle_nd_gt: " << AngleBetweenTwoVectors(n, d) << std::endl;
    std::cout << "Angle_nn: " << AngleBetweenTwoVectors(n_est, n) << std::endl;
    std::cout << "Angle_dd:" << AngleBetweenTwoVectors(d_est, d) << std::endl;
    std::cout << "Dist_nd_est: " << n_est.norm() / d_est.norm() << std::endl;
    std::cout << "Dist_nd_gt: " << n.norm() / d.norm() << std::endl;
}
