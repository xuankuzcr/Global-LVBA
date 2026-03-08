#ifndef UTILS_HPP
#define UTILS_HPP

#include <string>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <unordered_map>
#include <pcl/visualization/pcl_visualizer.h>
#include <opencv2/features2d.hpp>
#include <algorithm>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_types.h>
#include "sophus/se3.h"
#include <omp.h>

typedef pcl::PointXYZINormal PointType;
typedef pcl::PointXYZRGB PointTypeRGB;
typedef pcl::PointXYZRGBA PointTypeRGBA;
typedef pcl::PointCloud<PointType> PointCloudXYZI;
typedef std::vector<PointType, Eigen::aligned_allocator<PointType>> PointVector;
typedef pcl::PointCloud<PointTypeRGB> PointCloudXYZRGB;
typedef pcl::PointCloud<PointTypeRGBA> PointCloudXYZRGBA;

namespace sift {
    struct Keypoint {
        // discrete coordinates
        int i;
        int j;
        int octave;
        int scale; //index of gaussian image inside the octave

        // continuous coordinates (interpolated)
        float x;
        float y;
        float sigma;
        float extremum_val; //value of interpolated DoG extremum
        
        std::array<uint8_t, 128> descriptor;
    };
}

namespace lvba {

// ======================= 重投影（含畸变）+ 白化 =======================
// 模型：Brown–Conrady，参数顺序：k1,k2,p1,p2
// 白化：像素残差分别除以 su, sv（常用 su=sv=1.0 px 或 0.5 px）
struct ReprojErrorWhitenedDistorted {
    ReprojErrorWhitenedDistorted(double u, double v,
                                 double fx, double fy, double cx, double cy,
                                 double k1, double k2, double p1, double p2,
                                 double su = 1.0, double sv = 1.0)
    : u_(u), v_(v),
      fx_(fx), fy_(fy), cx_(cx), cy_(cy),
      k1_(k1), k2_(k2), p1_(p1), p2_(p2),
      su_(su), sv_(sv) {}
  
    template <typename T>
    bool operator()(const T* const q_cw,    // [w, x, y, z]
                    const T* const t_cw,    // [tx, ty, tz]
                    const T* const X_w,     // [X, Y, Z]
                    T* residuals) const {

      // 世界 -> 相机
      T Xc[3];
      {
        T Xw[3] = { X_w[0], X_w[1], X_w[2] };
        T RX[3];
        ceres::QuaternionRotatePoint(q_cw, Xw, RX);
        Xc[0] = RX[0] + t_cw[0];
        Xc[1] = RX[1] + t_cw[1];
        Xc[2] = RX[2] + t_cw[2];
      }

      if (Xc[2] <= T(1e-8)) { residuals[0] = T(0); residuals[1] = T(0); return true; };
  
      // 数值稳定：避免 z→0 或 z<=0
      T z = Xc[2];
    //   const T z_min = T(1e-8);
    //   z = z < z_min ? z_min : z;
  
      // 归一化坐标
      const T xn = Xc[0] / z;
      const T yn = Xc[1] / z;
  
      // Brown–Conrady 畸变
      const T r2 = xn*xn + yn*yn;
      const T r4 = r2 * r2;
  
      const T k1 = T(k1_), k2 = T(k2_);
      const T p1 = T(p1_), p2 = T(p2_);
  
      const T radial = T(1.0) + k1*r2 + k2*r4;
      const T x_tan  = T(2.0)*p1*xn*yn + p2*(r2 + T(2.0)*xn*xn);
      const T y_tan  = p1*(r2 + T(2.0)*yn*yn) + T(2.0)*p2*xn*yn;
  
      const T xdist = xn * radial + x_tan;
      const T ydist = yn * radial + y_tan;
  
      // 像素坐标
      const T u_pred = T(fx_) * xdist + T(cx_);
      const T v_pred = T(fy_) * ydist + T(cy_);
  
      // 白化（同一度量）
      residuals[0] = (u_pred - T(u_)) / T(su_);
      residuals[1] = (v_pred - T(v_)) / T(sv_);
      return true;
    }
  
    static ceres::CostFunction* Create(double u, double v,
                                       double fx, double fy, double cx, double cy,
                                       double k1, double k2, double p1, double p2,
                                       double su = 1.0, double sv = 1.0) {
      return (new ceres::AutoDiffCostFunction<ReprojErrorWhitenedDistorted, 2, 4, 3, 3>(
                new ReprojErrorWhitenedDistorted(u, v, fx, fy, cx, cy, k1, k2, p1, p2, su, sv)));
    }
  
    double u_, v_;
    double fx_, fy_, cx_, cy_;
    double k1_, k2_, p1_, p2_;
    double su_, sv_;
};

// ================ 点到平面距离 + 白化（1维残差） =================
// r = (n^T X + d) / sigma
struct PointPlaneErrorWhitened {
    PointPlaneErrorWhitened(const Eigen::Vector3d& n, double d, double sigma)
    : nx_(n.x()), ny_(n.y()), nz_(n.z()), d_(d), s_(std::max(1e-9, sigma)) {}
  
    template <typename T>
    bool operator()(const T* const X_w, T* residuals) const {
      T r = T(0.) - (T(nx_) * X_w[0] + T(ny_) * X_w[1] + T(nz_) * X_w[2] + T(d_));
      // residuals[0] = ceres::sqrt(r * r) / T(s_);
        residuals[0] = ceres::sqrt(r * r + 1e-12) / T(s_);
      return true;
    }
  
    static ceres::CostFunction* Create(const Eigen::Vector3d& n, double d, double sigma) {
      return (new ceres::AutoDiffCostFunction<PointPlaneErrorWhitened, 1, 3>(
                new PointPlaneErrorWhitened(n, d, sigma)));
    }
  
    double nx_, ny_, nz_, d_, s_;
};

// ---- 输出 Track 结构 ----
struct Track {
    Eigen::Vector3d Xw_fused = Eigen::Vector3d::Zero();           // 融合后的世界系3D
    std::vector<std::pair<int,int>> observations;                 // (image_id, kp_id)
    std::vector<int> inlier_indices;                              // 观测列表中被判为内点的下标
    // double rms = -1.0;
    // double mad = -1.0;
};

// ---- 小工具：像素→深度获取（支持 CV_32F / CV_16U），双线性插值 ----
inline bool fetchDepthBilinear(const cv::Mat& depth, float u, float v, float& d_out, float depth_scale=1.0f) {
    if (depth.empty()) return false;
    int w = depth.cols, h = depth.rows;
    if (u < 0.0f || v < 0.0f || u >= w-1 || v >= h-1) return false;
    int x = (int)std::floor(u);
    int y = (int)std::floor(v);
    float du = u - x;
    float dv = v - y;

    auto getD = [&](int xx, int yy)->float {
        if (depth.type() == CV_32FC1) {
            float d = depth.at<float>(yy, xx);
            return d;
        } else if (depth.type() == CV_16UC1) {
            uint16_t d = depth.at<uint16_t>(yy, xx);
            return depth_scale * static_cast<float>(d);
        } else {
            return -1.0f;
        }
    };
    float d00 = getD(x, y);
    float d10 = getD(x+1, y);
    float d01 = getD(x, y+1);
    float d11 = getD(x+1, y+1);
    if (d00 <= 0 || d10 <= 0 || d01 <= 0 || d11 <= 0) return false;

    d_out = (1-du)*(1-dv)*d00 + du*(1-dv)*d10 + (1-du)*dv*d01 + du*dv*d11;
    return d_out > 0.0f;
}

// ---- 像素+深度 -> 相机系3D（含去畸变，Brown-Conrady模型）----
inline Eigen::Vector3d backProjectCam(double u, double v, double d, 
                                       double fx, double fy, double cx, double cy,
                                       double k1, double k2, double p1, double p2) {
    // 1. 归一化畸变坐标
    double x_d = (u - cx) / fx;
    double y_d = (v - cy) / fy;

    // 2. 迭代去畸变
    double x_u = x_d;
    double y_u = y_d;

    const int max_iter = 10;
    const double eps = 1e-9;

    for (int i = 0; i < max_iter; ++i) {
        double r2 = x_u * x_u + y_u * y_u;
        double r4 = r2 * r2;

        double radial = 1.0 + k1 * r2 + k2 * r4;
        double x_tan = 2.0 * p1 * x_u * y_u + p2 * (r2 + 2.0 * x_u * x_u);
        double y_tan = p1 * (r2 + 2.0 * y_u * y_u) + 2.0 * p2 * x_u * y_u;

        double x_new = (x_d - x_tan) / radial;
        double y_new = (y_d - y_tan) / radial;

        if (std::abs(x_new - x_u) < eps && std::abs(y_new - y_u) < eps) {
            x_u = x_new;
            y_u = y_new;
            break;
        }

        x_u = x_new;
        y_u = y_new;
    }

    // 3. 使用去畸变的归一化坐标反投影到3D
    return Eigen::Vector3d(x_u * d, y_u * d, d);
}

// ---- 相机->世界：给定 Rcw, tcw (世界->相机)，则 Rwc=Rcw^T, twc = -Rcw^T*tcw ----
inline Eigen::Vector3d camToWorld(const Eigen::Vector3d& Xc,
                                  const Eigen::Matrix3d& Rcw,
                                  const Eigen::Vector3d& tcw) {
    Eigen::Matrix3d Rwc = Rcw.transpose();
    Eigen::Vector3d twc = - Rwc * tcw;
    return Rwc * Xc + twc;
}

// ---- (i,j) -> all_matches_ 索引（按 0-1,0-2,...;1-2,1-3,... 顺序） ----
inline size_t pairIndex(int i, int j, int N) {
    // 前 i 轮已出现的配对数： (N-1) + (N-2) + ... + (N-1-i) = i*(2N-i-1)/2
    // 当前轮的偏移： (j - (i+1))
    // 要求 j > i
    return static_cast<size_t>( i*(2*N - i - 1)/2 + (j - i - 1) );
}

// ---- MAD 计算（中值绝对偏差，鲁棒尺度）----
inline double computeMAD(const std::vector<double>& resid) {
    if (resid.empty()) return -1.0;
    std::vector<double> r = resid;
    std::nth_element(r.begin(), r.begin()+r.size()/2, r.end());
    double med = r[r.size()/2];
    for (auto& x : r) x = std::abs(x - med);
    std::nth_element(r.begin(), r.begin()+r.size()/2, r.end());
    double mad = r[r.size()/2] * 1.4826; // 正态一致性
    return mad;
}

// 仅修改此函数：按你提出的4条规则筛 inliers；否则 inliers 留空
void pickLargestClusterAsInliers(
    const std::vector<Eigen::Vector3d>& points3d,
    const std::vector<int>& idx_valid,
    std::vector<int>& inliers)
{
    inliers.clear();
    const double kWithin = 0.1;
    const double kWithin2 = kWithin * kWithin;

    if (idx_valid.empty()) return;
    if (idx_valid.size() == 1) { inliers = idx_valid; return; }

    // ---- 快速全体一致性检查：包围盒快速判定 + 必要时早停两两 ----
    auto allPairwiseWithinFast = [&](const std::vector<int>& ids)->bool {
        if (ids.size() <= 1) return true;

        // 1) O(N) 包围盒
        Eigen::Vector3d mn( std::numeric_limits<double>::infinity(),
                            std::numeric_limits<double>::infinity(),
                            std::numeric_limits<double>::infinity());
        Eigen::Vector3d mx(-std::numeric_limits<double>::infinity(),
                           -std::numeric_limits<double>::infinity(),
                           -std::numeric_limits<double>::infinity());
        for (int id : ids) {
            const auto& p = points3d[id];
            mn = mn.cwiseMin(p);
            mx = mx.cwiseMax(p);
        }
        Eigen::Vector3d span = mx - mn;
        // 快速“必不满足”：任一轴跨度 > 阈值，则必有一对 > 阈值
        if (span.x() > kWithin || span.y() > kWithin || span.z() > kWithin) return false;
        // 快速“必满足”：对角线 <= 阈值，则任意两点距离 <= 阈值
        if (span.squaredNorm() <= kWithin2) return true;

        // 2) 模糊区再做两两检查（早停）
        const size_t M = ids.size();
        for (size_t i = 0; i + 1 < M; ++i) {
            const auto& pi = points3d[ids[i]];
            for (size_t j = i + 1; j < M; ++j) {
                const auto& pj = points3d[ids[j]];
                if ((pi - pj).squaredNorm() > kWithin2) return false; // 早停
            }
        }
        return true; // 没发现超阈值对
    };

    // 记录开始时间
    // auto start = std::chrono::steady_clock::now();

    // 规则1：全体两两 <= 0.1m
    if (allPairwiseWithinFast(idx_valid)) {
        inliers = idx_valid;
        return;
    }
    // 记录结束时间
    // auto end = std::chrono::steady_clock::now();
    // std::chrono::duration<double> elapsed_seconds = end - start;
    // std::cout << "Time taken for allPairwiseWithinFast: " << elapsed_seconds.count() << "s\n";
    

    // 记录开始时间
    // start = std::chrono::steady_clock::now();
    // ---- k=2 KMeans（快速版：10次迭代 + 平方距离）----
    std::vector<int> A, B;
    {
        // 远点初始化
        Eigen::Vector3d c0 = points3d[idx_valid.front()];
        int far_id = idx_valid.front();
        double best = -1.0;
        for (int id : idx_valid) {
            double d2 = (points3d[id] - c0).squaredNorm();
            if (d2 > best) { best = d2; far_id = id; }
        }
        Eigen::Vector3d c1 = points3d[far_id];

        auto center_of = [&](const std::vector<int>& ids){
            Eigen::Vector3d m = Eigen::Vector3d::Zero();
            for (int id : ids) m += points3d[id];
            if (!ids.empty()) m /= static_cast<double>(ids.size());
            return m;
        };

        const int max_iters = 10;     // 更小的迭代上限
        const double tol = 1e-5;      // 略放宽收敛阈值
        for (int it = 0; it < max_iters; ++it) {
            A.clear(); B.clear();
            for (int id : idx_valid) {
                const auto& p = points3d[id];
                double d0 = (p - c0).squaredNorm();
                double d1 = (p - c1).squaredNorm();
                (d0 <= d1 ? A : B).push_back(id);
            }
            if (A.empty() && !B.empty()) { A.push_back(B.back()); B.pop_back(); }
            else if (!A.empty() && B.empty()) { B.push_back(A.back()); A.pop_back(); }

            Eigen::Vector3d nc0 = center_of(A);
            Eigen::Vector3d nc1 = center_of(B);
            double shift = (nc0 - c0).norm() + (nc1 - c1).norm();
            c0 = nc0; c1 = nc1;
            if (shift < tol) break;
        }
    }

    // 规则2/3：先看大簇，再看小簇（都用快速两两检查）
    const std::vector<int>* bigger = &A;
    const std::vector<int>* smaller = &B;
    if (B.size() > A.size()) { bigger = &B; smaller = &A; }

    if (!bigger->empty() && allPairwiseWithinFast(*bigger)) { inliers = *bigger; return; }
    if (!smaller->empty() && allPairwiseWithinFast(*smaller)) { inliers = *smaller; return; }
    // 记录结束时间
    // end = std::chrono::steady_clock::now();
    // elapsed_seconds = end - start;
    // std::cout << "Time taken for kmeans: " << elapsed_seconds.count() << "s\n";

    // 规则4：都不满足
    // 规则4：都不满足
    inliers.clear();
}

static inline cv::Mat preprocessLowTextureBGR(const cv::Mat& bgr, bool enable_x2_upsample)
{
    cv::Mat img = bgr.clone();
    // 可选：×2 上采样（对白墙弱纹理很有帮助）
    if (enable_x2_upsample) {
        cv::resize(img, img, cv::Size(img.cols*2, img.rows*2), 0, 0, cv::INTER_CUBIC);
    }

    // 在 Lab 的 L 通道做 CLAHE，再轻度反锐化
    cv::Mat lab; cv::cvtColor(img, lab, cv::COLOR_BGR2Lab);
    std::vector<cv::Mat> ch; cv::split(lab, ch);
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8)); // clip=2.0, tile=8x8
    clahe->apply(ch[0], ch[0]);
    cv::merge(ch, lab);
    cv::cvtColor(lab, img, cv::COLOR_Lab2BGR);

    // 轻度 unsharp mask（半径≈1，amount≈0.5）
    cv::Mat blur; cv::GaussianBlur(img, blur, cv::Size(0,0), 1.0);
    cv::addWeighted(img, 1.5, blur, -0.5, 0, img);
    return img;
}

template <typename Scalar>
inline Eigen::Matrix<Scalar, 3, 3> EulerToRot(Scalar roll, Scalar pitch, Scalar yaw) {
    using Mat3 = Eigen::Matrix<Scalar, 3, 3>;
    using Vec3 = Eigen::Matrix<Scalar, 3, 1>;

    Eigen::AngleAxis<Scalar> Rx(roll,  Vec3::UnitX());
    Eigen::AngleAxis<Scalar> Ry(pitch, Vec3::UnitY());
    Eigen::AngleAxis<Scalar> Rz(yaw,   Vec3::UnitZ());

    return (Rz * Ry * Rx).toRotationMatrix();
}

}

static inline bool parseTimestampFromName(const std::string& fname, double& ts_out)
{
    // 匹配以数字（可带小数）开头的时间戳：如 0.423131.png / 12.5.pcd
    // 也兼容文件名前面有路径、后缀等，找第一个浮点数字串
    static const std::regex re(R"(([0-9]+(?:\.[0-9]+)?))");
    std::smatch m;
    if (std::regex_search(fname, m, re)) {
        try {
            ts_out = std::stod(m[1].str());
            return true;
        } catch (...) {
            return false;
        }
    }
    return false;
}

// template <typename PointT>
inline void transformPointBodyToWorld(pcl::PointCloud<PointType>::Ptr cloud, const Sophus::SE3& T_wb)
{
  const Eigen::Matrix3d R_wb = T_wb.rotation_matrix();
  const Eigen::Vector3d t_wb = T_wb.translation();

  for (auto& pt : cloud->points) {
    const Eigen::Vector3d p_b(pt.x, pt.y, pt.z);
    const Eigen::Vector3d p_w = R_wb * p_b + t_wb;
    pt.x = static_cast<float>(p_w.x());
    pt.y = static_cast<float>(p_w.y());
    pt.z = static_cast<float>(p_w.z());
  }
}

static inline std::string fmtTs(double ts, int precision = 6) {
    std::ostringstream oss;
    oss.setf(std::ios::fixed);
    oss << std::setprecision(precision) << ts;
    return oss.str();
}

void printProgressBar(size_t current, size_t total, size_t barWidth = 50) {
    if (total == 0) return;
    double progress = static_cast<double>(current) / static_cast<double>(total);
    size_t pos = static_cast<size_t>(barWidth * progress);

    std::cout << "[";
    for (size_t i = 0; i < barWidth; ++i) {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }
    std::cout << "] " << std::setw(3) << int(progress * 100.0) << "% ("
              << current << "/" << total << ")\r";  // 注意 \r 回到行首
    std::cout.flush();
}

#endif // UTILS_HPP