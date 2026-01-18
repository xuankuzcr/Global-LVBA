#ifndef LVBA_SYSTEM_H
#define LVBA_SYSTEM_H

#include <ros/ros.h>
#include <thread>
#include <pcl/common/common.h>
#include <GL/glew.h>   // 先
#include <GL/gl.h>     // 后
#include <pcl/filters/voxel_grid.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include "dataset_io.h"
#include "sophus/se3.h"
#include "SiftGPU.h"
#include <sqlite3.h>
#include <sstream>
#include <utility>

#include "BALM/tools.hpp"
#include "BALM/bavoxel.hpp"

namespace fs = std::filesystem;

namespace lvba {

class LvbaSystem {
public:
    LvbaSystem(ros::NodeHandle& nh);
    ~LvbaSystem() = default;

    void runFullPipeline();

    void initFromDatasetIO();

    void runWindowBA(const std::vector<IMUST>& x_buf_full,
                     const std::vector<pcl::PointCloud<PointType>::Ptr>& pl_fulls_full,
                     std::vector<IMUST>& anchor_poses,
                     std::vector<pcl::PointCloud<PointType>::Ptr>& anchor_clouds);

    void runLidarBA();
    void runVisualBAWithLidarAssist();

    void saveTrackFeaturesOnImages();

    void extractAndMatchFeaturesGPU();

    bool loadFromColmapDB();

    void generateDepthWithVoxel();

    void updateCameraPosesFromLidar();

    void BuildTracksAndFuse3D();

    void visualizeProj();

    void buildGridMapFromOptimized();

    void showTracksComparePCL();

    void optimizeCameraPoses();

    void drawAndSaveMatchesGPU(
        const std::string& out_dir,
        int id1, int id2,
        const cv::Mat& img1, const cv::Mat& img2,
        const std::vector<SiftGPU::SiftKeypoint>& kpts1,
        const std::vector<SiftGPU::SiftKeypoint>& kpts2,
        const std::vector<std::pair<int,int>>& matches);

    void VisualizeOptComparison(
            const std::vector<double>& image_ids,
            bool save_merged_pcd,
            const std::string& merged_pcd_path);

    void visialTrackCloud();

    std::string getImagePath(double image_id);
    std::string getPcdPath(double pcd_id);

    void pubRGBCloud();

    ros::Publisher pub_path_, pub_test_, pub_show_, pub_cute_, pub_cloud_before_, pub_cloud_after_, pub_cloud_map_;
    void data_show(vector<IMUST> x_buf, vector<pcl::PointCloud<PointType>::Ptr> &pl_fulls);
    template <typename T> void pub_pl_func(T &pl, ros::Publisher &pub);
    
    VOX_HESS *voxhess;
    BALM2 *opt_lsv;


    ros::NodeHandle& nh_;
    DatasetIOPtr dataset_io_;
    ros::Publisher cloud_pub_after_;
    ros::Publisher cloud_pub_before_;

    std::unordered_map<VOXEL_LOC, OCTO_TREE_ROOT*> surf_map;

    inline bool ProjectToImage(
        const Eigen::Matrix3d& Rcw, const Eigen::Vector3d& tcw,
        const Eigen::Vector3d& Xw,
        double* u, double* v, double* Zc) const;

    std::vector<double> images_ids_;
    std::vector<Sophus::SE3> poses_before_;
    std::vector<Sophus::SE3> poses_;  // 当前使用的相机位姿（同步/优化后）

    std::vector<std::pair<double, double>> image_pairs_;
    std::vector<std::pair<Sophus::SE3, Sophus::SE3>> pose_pairs_;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr pub_cloud_, pub_cloud_b_;

    std::vector<std::vector<sift::Keypoint>> all_keypoints_; // 存储所有图像的特征点
    std::vector<std::vector<std::pair<int, int>>> all_matches_; // 存储所有图像对的匹配结果
    std::vector<std::vector<VOXEL_LOC>> all_voxel_ids_;
    std::vector<std::vector<std::vector<std::pair<int,int>>>> adj_;
    std::vector<Eigen::Vector3d> points3d_; // 存储所有共视三维点

    std::unordered_map<double, int> ts2idx;


    std::unordered_map<VOXEL_LOC, std::vector<Eigen::Vector3d>> grid_map_;

    std::vector<cv::Mat> all_depths_;
    std::vector<Track> tracks_before_, tracks_;
    std::string dataset_path_;

    std::vector<IMUST> rel_poses_to_anchor_; //每一帧在WIN_BA优化后相对于窗口锚点帧的相对位姿
    std::vector<int> anchor_index_per_frame_; //用来表示每一帧对应的锚点帧下标
    std::vector<IMUST> optimized_x_buf_;  //优化后的所有帧在世界系下的新状态会重新放进成员变量x_buf_ 里边

    bool enable_visual_ba_, enable_lidar_ba_;

    int image_width_, image_height_;
    double fx_, fy_, cx_, cy_, d0_, d1_, d2_, d3_, scale_;
    Eigen::Matrix3d Rli_, Rcl_, Rci_, Rcw_;
    Eigen::Vector3d tli_, tcl_, tci_, tcw_;

    std::vector<Eigen::Matrix3d> Rcw_all_, Rcw_all_optimized_;
    std::vector<Eigen::Vector3d> tcw_all_, tcw_all_optimized_;

    int obser_thr_ = 3;
    std::ofstream fout_points_after, fout_points_before, fout_poses_after, fout_poses_before;

    double min_view_angle_deg_, reproj_mean_thr_px_, filter_size_points3D_;

    bool colmap_output_enable_;
};

}
#endif // LVBA_SYSTEM_H
