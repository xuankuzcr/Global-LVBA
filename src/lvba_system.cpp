#include "lvba_system.h"

namespace lvba {
    
LvbaSystem::LvbaSystem(ros::NodeHandle& nh) : nh_(nh)                                
{
    dataset_io_.reset(new DatasetIO(nh_));

    cloud_pub_after_ = nh_.advertise<sensor_msgs::PointCloud2>("/lvba/cloud_after", 1, true);
    cloud_pub_before_ = nh_.advertise<sensor_msgs::PointCloud2>("/lvba/cloud_before", 1, true);
    pub_test_ = nh_.advertise<sensor_msgs::PointCloud2>("/map_test", 100);
    pub_path_ = nh_.advertise<sensor_msgs::PointCloud2>("/map_path", 100);
    pub_show_ = nh_.advertise<sensor_msgs::PointCloud2>("/map_show", 100);
    pub_cute_ = nh_.advertise<sensor_msgs::PointCloud2>("/map_cute", 100);

    pub_cloud_before_ = nh_.advertise<sensor_msgs::PointCloud2>("viz/cloud_before", 1, true);
    pub_cloud_after_  = nh_.advertise<sensor_msgs::PointCloud2>("viz/cloud_after", 1, true);

    nh_.param<bool>("data_config/enable_lidar_ba", enable_lidar_ba_, true);
    nh_.param<bool>("data_config/enable_visual_ba", enable_visual_ba_, true);
    nh_.param<double>("track_fusion/min_view_angle", min_view_angle_deg_, 8.0);
    nh_.param<double>("track_fusion/reproj_mean_thr", reproj_mean_thr_px_, 3.0);

    nh_.param<bool>("colmap_output/enable", colmap_output_enable_, true);
    nh_.param<double>("colmap_output/filter_size_points3D", filter_size_points3D_, 0.01);
}

void LvbaSystem::runFullPipeline() 
{
    initFromDatasetIO();
    if(enable_lidar_ba_) runLidarBA();
    if(enable_visual_ba_) runVisualBAWithLidarAssist();
    ros::spin();
}

void LvbaSystem::runVisualBAWithLidarAssist()
{
    buildGridMapFromOptimized();
    updateCameraPosesFromLidar();
    generateDepthWithVoxel();
    extractAndMatchFeaturesGPU();
    BuildTracksAndFuse3D();
    optimizeCameraPoses();
    visualizeProj();
    pubRGBCloud();
}

template <typename T>
void LvbaSystem::pub_pl_func(T &pl, ros::Publisher &pub)
{
  pl.height = 1; pl.width = pl.size();
  sensor_msgs::PointCloud2 output;
  pcl::toROSMsg(pl, output);
  output.header.frame_id = "map";
  output.header.stamp = ros::Time::now();
  pub.publish(output);
}

void LvbaSystem::data_show(vector<IMUST> x_buf, vector<pcl::PointCloud<PointType>::Ptr> &pl_fulls)
{
  IMUST es0 = x_buf[0];
  for(uint i=0; i<x_buf.size(); i++)
  {
    x_buf[i].p = es0.R.transpose() * (x_buf[i].p - es0.p);
    x_buf[i].R = es0.R.transpose() * x_buf[i].R;
  }

  pcl::PointCloud<PointType> pl_send, pl_path;
  int winsize = x_buf.size();
  for(int i=0; i<winsize; i++)
  {
    pcl::PointCloud<PointType> pl_tem = *pl_fulls[i];
    down_sampling_voxel(pl_tem, 0.05);
    pl_transform(pl_tem, x_buf[i]);
    pl_send += pl_tem;

    // if((i%2==0 && i!=0) || i == winsize-1)
    // {
    //   pub_pl_func(pl_send, pub_show_);
    //   pl_send.clear();
    //   sleep(0.3);
    // }

    PointType ap;
    ap.x = x_buf[i].p.x();
    ap.y = x_buf[i].p.y();
    ap.z = x_buf[i].p.z();
    ap.curvature = i;
    pl_path.push_back(ap);
  }
  down_sampling_voxel(pl_send, 0.05);
  pub_pl_func(pl_send, pub_show_);
  pub_pl_func(pl_path, pub_path_);
}

void LvbaSystem::runWindowBA(const std::vector<IMUST>& x_buf_full,
                             const std::vector<pcl::PointCloud<PointType>::Ptr>& pl_fulls_full,
                             std::vector<IMUST>& anchor_poses,
                             std::vector<pcl::PointCloud<PointType>::Ptr>& anchor_clouds)
{
    anchor_poses.clear();
    anchor_clouds.clear();

    const bool run_window = dataset_io_->window_ba_enable_;
    const int window_size = dataset_io_->window_ba_size_;
    const double anchor_leaf = dataset_io_->anchor_leaf_size_;
    const bool use_window_ba_rel = dataset_io_->use_window_ba_rel_;
    const int total_size = static_cast<int>(x_buf_full.size());

    if (!run_window) {
        anchor_poses = x_buf_full;
        anchor_clouds = pl_fulls_full;
        for (int i = 0; i < total_size; ++i) {
            anchor_index_per_frame_[i] = i;
            rel_poses_to_anchor_[i].setZero(); // identity
        }
        return;
    }

    printf("[WindowBA] Running Window LiDAR BA, window size=%d, anchor leaf=%.2f ...\n", window_size, anchor_leaf);

    int win_total = 0;
    int win_skipped = 0;
    for (int start = 0; start < total_size; start += window_size) {
        int end = std::min(start + window_size, total_size);
        printProgressBar(end, total_size);

        int curr_win = end - start;
        if (curr_win <= 0) break;
        ++win_total;

        std::vector<IMUST> x_win(x_buf_full.begin() + start, x_buf_full.begin() + end);
        std::vector<IMUST> x_win_odom = x_win;
        std::vector<IMUST> x_win_aligned = x_win;
        std::vector<pcl::PointCloud<PointType>::Ptr> pl_win;
        pl_win.reserve(curr_win);
        for (int i = start; i < end; ++i) pl_win.push_back(pl_fulls_full[i]);

        std::unordered_map<VOXEL_LOC, OCTO_TREE_ROOT*> surf_map;
        for (int j = 0; j < curr_win; ++j) {
            cut_voxel(surf_map, *pl_win[j], x_win[j], j, curr_win,
                      dataset_io_->stage1_root_voxel_size_, dataset_io_->stage1_eigen_ratio_array_[0]);
        }

        std::unique_ptr<BALM2> opt_lsv(new BALM2(curr_win));
        std::unique_ptr<VOX_HESS> voxhess(new VOX_HESS(curr_win));
        for (auto iter = surf_map.begin(); iter != surf_map.end() && nh_.ok(); ++iter) {
            iter->second->recut(x_win);
            iter->second->tras_opt(*voxhess);
        }
        if (voxhess->plvec_voxels.size() < static_cast<size_t>(3 * x_win.size())) {
            for (auto& kv : surf_map) delete kv.second;
            ++win_skipped;
            continue;
        }
        opt_lsv->damping_iter(x_win, *voxhess);

        for (auto& kv : surf_map) delete kv.second;

        if (use_window_ba_rel && !x_win.empty()) {
            const IMUST& odom0 = x_win_odom[0];
            const IMUST& opt0  = x_win[0];
            Eigen::Matrix3d R_align = odom0.R * opt0.R.transpose();
            Eigen::Vector3d p_align = odom0.p - R_align * opt0.p;
            for (int j = 0; j < curr_win; ++j) {
                x_win_aligned[j].R = R_align * x_win[j].R;
                x_win_aligned[j].p = R_align * x_win[j].p + p_align;
            }
        } else {
            x_win_aligned = x_win_odom;
        }

        pcl::PointCloud<PointType>::Ptr merged(new pcl::PointCloud<PointType>());
        const IMUST anchor_pose = x_win_odom[0];
        const int anchor_idx = static_cast<int>(anchor_poses.size());
        for (int j = 0; j < curr_win; ++j) {
            pcl::PointCloud<PointType> tmp = *pl_win[j];
            IMUST rel;
            rel.R = anchor_pose.R.transpose() * x_win_aligned[j].R;
            rel.p = anchor_pose.R.transpose() * (x_win_aligned[j].p - anchor_pose.p);
            pl_transform(tmp, rel);
            *merged += tmp;

            const int global_idx = start + j;
            if (global_idx < total_size) {
                rel_poses_to_anchor_[global_idx] = rel;
                anchor_index_per_frame_[global_idx] = anchor_idx;
            }
        }
        down_sampling_voxel2(*merged, anchor_leaf);

        anchor_poses.push_back(anchor_pose);
        anchor_clouds.push_back(merged);
    }
    std::cout << std::endl;

    if (win_total > 0) {
        printf("[WindowBA] skipped %d/%d windows (%.2f%%)\n",
               win_skipped, win_total,
               100.0 * static_cast<double>(win_skipped) / static_cast<double>(win_total));
    }
}

void LvbaSystem::runLidarBA() 
{
    std::vector<IMUST> x_buf_full = dataset_io_->x_buf_;
    std::vector<pcl::PointCloud<PointType>::Ptr> pl_fulls_full = dataset_io_->pl_fulls_;
    const int total_size = static_cast<int>(x_buf_full.size());
    if (total_size == 0) {
        ROS_WARN("No poses in buffer, skip runLidarBA.");
        return;
    }

    data_show(x_buf_full, pl_fulls_full);
    printf("If no problem, input '1' to continue or '0' to exit...\n");
    int cont_flag = 1;
    std::cin >> cont_flag;
    if (cont_flag == 0) {
        return;
    }
    std::vector<IMUST> anchor_poses;
    std::vector<pcl::PointCloud<PointType>::Ptr> anchor_clouds;

    rel_poses_to_anchor_.assign(total_size, IMUST());
    anchor_index_per_frame_.assign(total_size, -1); //用来表示每一原始针对应的锚点帧的下标

    runWindowBA(x_buf_full, pl_fulls_full, anchor_poses, anchor_clouds);

    const int win_size = static_cast<int>(anchor_poses.size());
    const char* pass_name[2] = {"Stage 1", "Stage 2"};
    bool run_stage1 = dataset_io_->stage1_enable_;

    double root_voxel_size[2];
    std::array<float, 4> eigen_ratio_array[2];

    root_voxel_size[0] = dataset_io_->stage1_root_voxel_size_;
    root_voxel_size[1] = dataset_io_->stage2_root_voxel_size_;
    eigen_ratio_array[0].fill(0.f);
    eigen_ratio_array[1].fill(0.f);

    for (size_t i = 0; i < eigen_ratio_array[0].size(); ++i) {
        if (i < dataset_io_->stage1_eigen_ratio_array_.size())
            eigen_ratio_array[0][i] = dataset_io_->stage1_eigen_ratio_array_[i];
        if (i < dataset_io_->stage2_eigen_ratio_array_.size())
            eigen_ratio_array[1][i] = dataset_io_->stage2_eigen_ratio_array_[i];
    }

    int start_idx = run_stage1 ? 0 : 1;
    for (int idx = start_idx; idx < 2; ++idx) {
        cout << "[runLidarBA] Global LiDAR BA start... " << pass_name[idx] << endl;

        set_eigen_ratio_array(eigen_ratio_array[idx]);
        std::unordered_map<VOXEL_LOC, OCTO_TREE_ROOT*> surf_map;
        pcl::PointCloud<PointType> pl_send;
        pub_pl_func(pl_send, pub_show_);

        float eigen_ratio_val = eigen_ratio_array[idx][0];
        for (int j = 0; j < win_size; j++) {
            cut_voxel(surf_map, *anchor_clouds[j], anchor_poses[j], j, win_size,
                      root_voxel_size[idx], eigen_ratio_val);
        }

        std::unique_ptr<BALM2> opt_lsv(new BALM2(win_size));
        std::unique_ptr<VOX_HESS> voxhess(new VOX_HESS(win_size));

        for (auto iter = surf_map.begin(); iter != surf_map.end() && nh_.ok(); ++iter) {
            iter->second->recut(anchor_poses);
            iter->second->tras_opt(*voxhess);
            iter->second->tras_display(pl_send, anchor_poses, 0);
        }

        down_sampling_voxel(pl_send, 0.05);
        pub_pl_func(pl_send, pub_cute_);

        pl_send.clear();
        pub_pl_func(pl_send, pub_cute_);

        opt_lsv->damping_iter(anchor_poses, *voxhess);

        for (auto& kv : surf_map) delete kv.second;
    }

    cout << "[runLidarBA] Global BALM Finish..." << endl;

    optimized_x_buf_ = x_buf_full;
    for (size_t idx = 0; idx < optimized_x_buf_.size(); ++idx) {
        const int anchor_idx = (idx < anchor_index_per_frame_.size()) ? anchor_index_per_frame_[idx] : -1;
        if (anchor_idx < 0 || anchor_idx >= static_cast<int>(anchor_poses.size())) continue;

        const IMUST& rel = rel_poses_to_anchor_[idx];
        const IMUST& anchor = anchor_poses[anchor_idx];

        IMUST& out = optimized_x_buf_[idx];
        out.R = anchor.R * rel.R;
        out.p = anchor.R * rel.p + anchor.p;
    }

    dataset_io_->x_buf_ = optimized_x_buf_;
    
    // data_show(anchor_poses, anchor_clouds);
    data_show(dataset_io_->x_buf_, dataset_io_->pl_fulls_);
}

void LvbaSystem::updateCameraPosesFromLidar()
{
    const auto& lidar_opt = dataset_io_->x_buf_;
    const auto& lidar_orig = dataset_io_->x_buf_before_;
    const auto& cam_orig = dataset_io_->image_poses_;

    poses_.clear();
    poses_.reserve(cam_orig.size());

    std::vector<double> ts;
    ts.reserve(lidar_opt.size());
    for (const auto& x : lidar_opt) ts.push_back(x.t);

    for (size_t i = 0; i < images_ids_.size(); ++i) {
        double t_img = images_ids_[i];

        auto it = std::lower_bound(ts.begin(), ts.end(), t_img);
        size_t idx = (it == ts.end()) ? ts.size() - 1 : static_cast<size_t>(it - ts.begin());
        if (it != ts.begin() && it != ts.end()) {
            size_t prev = idx - 1;
            if (std::abs(ts[prev] - t_img) < std::abs(ts[idx] - t_img)) idx = prev;
        }
        if (idx >= lidar_opt.size() || idx >= lidar_orig.size()) {
            poses_.push_back(cam_orig[i]);
            continue;
        }

        Sophus::SE3 T_opt(lidar_opt[idx].R, lidar_opt[idx].p);
        Sophus::SE3 T_orig(lidar_orig[idx].R, lidar_orig[idx].p);
        Sophus::SE3 T_delta = T_opt * T_orig.inverse();

        Sophus::SE3 T_cam_new = T_delta * cam_orig[i];
        poses_.push_back(T_cam_new);
    }
}

void LvbaSystem::initFromDatasetIO() {

    dataset_path_  = dataset_io_->dataset_path_;
    images_ids_    = dataset_io_->images_ids_;
    poses_before_  = dataset_io_->image_poses_;
    // poses_         = dataset_io_->image_poses_;
    // cloud_         = dataset_io_->cloud_;
    all_voxel_ids_ = dataset_io_->all_voxel_ids_;

    if (images_ids_.size() != poses_before_.size()) {
        std::cerr << "Error: Number of images and poses do not match!" << std::endl;
        return;
    }

    for (size_t i = 0; i < images_ids_.size(); ++i) {
        for (size_t j = i + 1; j < images_ids_.size(); ++j) {
            image_pairs_.push_back(std::make_pair(images_ids_[i], images_ids_[j]));
        }
    }

    all_keypoints_.resize(images_ids_.size());
    all_depths_.reserve(images_ids_.size());

    image_width_  = dataset_io_->width_;
    image_height_ = dataset_io_->height_;
    scale_ = dataset_io_->resize_scale_;

    fx_ = dataset_io_->fx_;
    fy_ = dataset_io_->fy_;
    cx_ = dataset_io_->cx_;
    cy_ = dataset_io_->cy_;
    d0_ = dataset_io_->k1_;
    d1_ = dataset_io_->k2_;
    d2_ = dataset_io_->p1_;
    d3_ = dataset_io_->p2_; 

    std::vector<double> t_lidar2cam = dataset_io_->cameraextrinT_;
    std::vector<double> r_lidar2cam = dataset_io_->cameraextrinR_;

    tcl_ << t_lidar2cam[0], t_lidar2cam[1], t_lidar2cam[2];
    Rcl_ << r_lidar2cam[0], r_lidar2cam[1], r_lidar2cam[2],
            r_lidar2cam[3], r_lidar2cam[4], r_lidar2cam[5],
            r_lidar2cam[6], r_lidar2cam[7], r_lidar2cam[8];

    std::vector<double> t_lidar2imu = dataset_io_->extrinT_;
    std::vector<double> r_lidar2imu = dataset_io_->extrinR_;

    Eigen::Vector3d til; Eigen::Matrix3d Ril;
    til << t_lidar2imu[0], t_lidar2imu[1], t_lidar2imu[2];
    Ril << r_lidar2imu[0], r_lidar2imu[1], r_lidar2imu[2],
           r_lidar2imu[3], r_lidar2imu[4], r_lidar2imu[5],
           r_lidar2imu[6], r_lidar2imu[7], r_lidar2imu[8];

    Rli_ = Ril.transpose();
    tli_ = -Rli_ * til;
    Rci_ = Rcl_ * Rli_;
    tci_ = Rcl_ * tli_ + tcl_;
}


// 不需要包含 <GL/gl.h>，避免 GL_LUMINANCE / GL_UNSIGNED_BYTE 冲突
// 但要确保项目已链接 DevIL、GLEW/GLUT（SiftGPU 依赖）
bool LvbaSystem::loadFromColmapDB()
{
    constexpr uint64_t kColmapMaxNumImages = (1ull << 31) - 1;
    auto imageIdsToPairId = [kColmapMaxNumImages](uint32_t image_id1, uint32_t image_id2) -> uint64_t {
        if (image_id1 > image_id2) {
            std::swap(image_id1, image_id2);
        }
        return static_cast<uint64_t>(image_id1) * kColmapMaxNumImages +
               static_cast<uint64_t>(image_id2);
    };

    sqlite3* db = nullptr;
    if (sqlite3_open(dataset_io_->colmap_db_path_.c_str(), &db) != SQLITE_OK) {
        std::cerr << "[DB] open failed: " << sqlite3_errmsg(db) << "\n";
        return false;
    }

    // 1) 读取 images 表并将 image_id 和 file_name 对应起来
    std::unordered_map<std::string, uint32_t> name2id;
    size_t db_image_count = 0;
    {
        const char* sql = "SELECT image_id, name FROM images;";
        sqlite3_stmt* stmt = nullptr;
        if (sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr) == SQLITE_OK) {
            while (sqlite3_step(stmt) == SQLITE_ROW) {
                uint32_t id = (uint32_t)sqlite3_column_int64(stmt, 0);
                std::string name = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
                name2id[name] = id;
                name2id[fs::path(name).filename().string()] = id;
                ++db_image_count;
            }
        }
        sqlite3_finalize(stmt);
    }
    std::cout << "[DB] ColmapDB images count = " << db_image_count << "\n";

    if (db_image_count != images_ids_.size()) {
        std::cerr << "[DB] Warning: DB images count (" << db_image_count
                  << ") != dataset images count (" << images_ids_.size() << ")\n";
        // 构建新的 数据库
        std::cout << "[DB] Rebuilding COLMAP database...\n";
        sqlite3_close(db);
        return false;
    }

    ts2idx.clear();
    ts2idx.reserve(images_ids_.size());
    for (int i = 0; i < (int)images_ids_.size(); ++i) {
        ts2idx[images_ids_[i]] = i;
    }

    // 辅助函数：根据时间戳查找对应的图像 ID
    auto imageIdOfTs = [&](double ts)->int {
        std::string p = getImagePath(ts);
        std::string base = fs::path(p).filename().string();
        auto it1 = name2id.find(base);
        if (it1 != name2id.end()) return (int)it1->second;
        return -1;
    };

    // 2) 读取 keypoints 表并写入 all_keypoints_
    if ((int)all_keypoints_.size() < (int)images_ids_.size())
        all_keypoints_.resize(images_ids_.size());

    {
        const char* sql = "SELECT rows, cols, data FROM keypoints WHERE image_id=?;";
        sqlite3_stmt* stmt = nullptr;
        if (sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr) != SQLITE_OK) {
            std::cerr << "[DB] keypoints SELECT prepare failed.\n";
        } else {
            for (size_t i = 0; i < images_ids_.size(); ++i) {
                int image_id = imageIdOfTs(images_ids_[i]);
                if (image_id < 0) continue;

                sqlite3_reset(stmt);
                sqlite3_bind_int(stmt, 1, image_id);
                if (sqlite3_step(stmt) == SQLITE_ROW) {
                    int rows = sqlite3_column_int(stmt, 0);
                    int cols = sqlite3_column_int(stmt, 1);  // 4 或 6
                    const void* blob = sqlite3_column_blob(stmt, 2);
                    int blob_bytes = sqlite3_column_bytes(stmt, 2);

                    if (blob && blob_bytes == rows * cols * (int)sizeof(float)) {
                        const float* fp = reinterpret_cast<const float*>(blob);
                        auto& vec = all_keypoints_[i];
                        vec.resize(rows);
                        for (int r = 0; r < rows; ++r) {
                            sift::Keypoint kp{};
                            kp.x = fp[r * cols + 0];
                            kp.y = fp[r * cols + 1];
                            if (cols >= 3) kp.sigma = fp[r * cols + 2];
                            if (cols >= 4) kp.extremum_val = fp[r * cols + 3];
                            vec[r] = kp;
                        }
                    }
                }
            }
        }
        sqlite3_finalize(stmt);
    }

    // 3) 读取 two_view_geometries 表并写入 all_matches_（只读取内点匹配）
    all_matches_.assign(image_pairs_.size(), {});
    {
        const char* sql = "SELECT rows, cols, data FROM two_view_geometries WHERE pair_id=?;";
        sqlite3_stmt* stmt = nullptr;
        if (sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr) != SQLITE_OK) {
            std::cerr << "[DB] two_view_geometries SELECT prepare failed.\n";
        } else {
            for (size_t k = 0; k < image_pairs_.size(); ++k) {
                const double ts1 = image_pairs_[k].first;
                const double ts2 = image_pairs_[k].second;

                const int id1 = imageIdOfTs(ts1);   // DB 的 image_id
                const int id2 = imageIdOfTs(ts2);
                if (id1 < 0 || id2 < 0) continue;

                // 关键点容器索引
                int idx1 = ts2idx[ts1]; // images_ids_ 的顺序下标
                int idx2 = ts2idx[ts2];
                if (idx1 < 0 || idx1 >= (int)all_keypoints_.size() ||
                    idx2 < 0 || idx2 >= (int)all_keypoints_.size()) {
                    continue;
                }
                const auto& kpts1 = all_keypoints_[idx1];
                const auto& kpts2 = all_keypoints_[idx2];
                if (kpts1.empty() || kpts2.empty()) {
                    std::cout << "[DB] keypoints of (" << ts1 << "," << ts2 << ") is empty.\n";
                    continue;
                }

                // pair_id 计算已保证 image_id 升序
                const bool swapped = (id1 > id2);  // 仅用于索引方向校正
                const uint64_t pair_id = imageIdsToPairId((uint32_t)id1, (uint32_t)id2);

                sqlite3_reset(stmt);
                sqlite3_bind_int64(stmt, 1, (sqlite3_int64)pair_id);
                if (sqlite3_step(stmt) != SQLITE_ROW) continue;

                const int rows = sqlite3_column_int(stmt, 0);
                const int cols = sqlite3_column_int(stmt, 1); // 应为 2
                const void* blob = sqlite3_column_blob(stmt, 2);
                const int blob_bytes = sqlite3_column_bytes(stmt, 2);
                if (cols != 2 || !blob || rows <= 0 ||
                    blob_bytes != rows * 2 * (int)sizeof(uint32_t)) {
                    continue;
                }

                const uint32_t* up = reinterpret_cast<const uint32_t*>(blob);
                auto& vec = all_matches_[k];
                vec.reserve(rows);

                for (int r = 0; r < rows; ++r) {
                    int i_small = (int)up[2*r + 0];  // 索引对应 "较小 image_id" 那一侧
                    int i_large = (int)up[2*r + 1];  // 索引对应 "较大 image_id" 那一侧

                    // 若当前(ts1,ts2)的 id 顺序与 pair_id 的顺序不一致，则交换
                    int i1 = i_small;
                    int i2 = i_large;
                    if (swapped) std::swap(i1, i2);

                    // 避免 drawMatches 越界
                    if (i1 >= 0 && i1 < (int)kpts1.size() &&
                        i2 >= 0 && i2 < (int)kpts2.size()) {
                        vec.emplace_back(i1, i2);
                    }
                }
            }
        }
        sqlite3_finalize(stmt);
    }

    sqlite3_close(db);
    std::cout << "[DB] Loaded keypoints & inlier matches from " << dataset_io_->colmap_db_path_ << "\n";
    return true;
}

void LvbaSystem::extractAndMatchFeaturesGPU()
{
    // ts -> 顺序下标映射, 保持 all_keypoints_ 的按序存储
    std::unordered_map<double, int> ts2idx;
    ts2idx.reserve(images_ids_.size());
    for (int i = 0; i < (int)images_ids_.size(); ++i) ts2idx[images_ids_[i]] = i;

    if ((int)all_keypoints_.size() < (int)images_ids_.size())
        all_keypoints_.resize(images_ids_.size());

    if (loadFromColmapDB()) {
        std::cout << "[Frontend] Using existing COLMAP DB: "
                  << dataset_io_->colmap_db_path_ << "\n";
        return;  // 已经拿到结果，直接返回
    }

    // 初始化 SiftGPU
    SiftGPU sift;
    const char* argv[] = {"-fo","-1","-loweo","-w","3","-t","0.01","-e","12","-v","0"};
    sift.ParseParam(10, const_cast<char**>(argv));
    if (sift.CreateContextGL() != SiftGPU::SIFTGPU_FULL_SUPPORTED) {
        std::cerr << "[SiftGPU] Not supported or context creation failed.\n";
        return;
    }

    // 匹配器
    SiftMatchGPU matcher;
    matcher.VerifyContextGL();
    // const float kRatioMax  = 1.0f;  // Lowe 比值阈值（越小越严格）0.85
    // const float kDistMax   = 0.5f;  // 距离上限（按 SiftGPU 自己的度量，先给个中性值）
    // const int   kUseMBM    = 1;      // mutual-best-match（双向最邻近）
    // matcher.SetRatio(0.8f);

    // 缓存：按时间戳提取一次
    struct ImgCache {
        std::vector<float> desc;                  // 128*N
        std::vector<SiftGPU::SiftKeypoint> kpt;   // GPU关键点
        std::vector<sift::Keypoint> kpt_conv;     // 你的Keypoint（至少x,y）
    };
    std::unordered_map<double, ImgCache> cache;
    cache.reserve(image_pairs_.size() * 2);

    auto extract_once = [&](double ts) -> bool {
        if (cache.find(ts) != cache.end()) return true;

        std::string path = getImagePath(ts);
        cv::Mat img = cv::imread(path, cv::IMREAD_COLOR);
        // cv::Mat img = preprocessLowTextureBGR(img0, false);

        if (img.cols != image_width_ || img.rows != image_height_) 
        {
            cv::resize(img, img, cv::Size(img.cols * scale_, img.rows * scale_), 0, 0, cv::INTER_LINEAR);
        }

        if (img.empty()) {
            std::cerr << "[SiftGPU] imread failed: " << path << "\n";
            return false;
        }

        // OpenCV 为 BGR
        if (!sift.RunSIFT(img.cols, img.rows, img.data, GL_BGR, GL_UNSIGNED_BYTE)) {
            std::cerr << "[SiftGPU] RunSIFT failed for " << path << "\n";
            return false;
        }

        int n = sift.GetFeatureNum();
        ImgCache entry;
        entry.desc.resize(128 * n);
        entry.kpt.resize(n);
        if (n > 0) sift.GetFeatureVector(entry.kpt.data(), entry.desc.data());

        // 转成你的 Keypoint（至少 x,y）
        entry.kpt_conv.resize(n);
        for (int i = 0; i < n; ++i) {
            sift::Keypoint kp{};
            kp.x = entry.kpt[i].x;
            kp.y = entry.kpt[i].y;
            // 如需：kp.scale = entry.kpt[i].s; kp.orientation = entry.kpt[i].o;
            entry.kpt_conv[i] = kp;
        }

        // 写回 all_keypoints_（按顺序下标）
        auto it = ts2idx.find(ts);
        if (it != ts2idx.end()) {
            int idx = it->second;
            if (all_keypoints_[idx].empty())
                all_keypoints_[idx] = entry.kpt_conv;
        }

        cache.emplace(ts, std::move(entry));
        return true;
    };

    // 遍历时间戳图像对并匹配
    all_matches_.clear();
    all_matches_.resize(image_pairs_.size());
    std::vector<bool> is_bad_pair(image_pairs_.size(), false);
    int pair_count = 0;
    std::cout << "[SiftGPU] Start Matching " << image_pairs_.size() << " image pairs ...\n";
    for (const auto& pr : image_pairs_) {
        double ts1 = pr.first, ts2 = pr.second;
        if (!extract_once(ts1) || !extract_once(ts2)) {
            std::cout << "[SiftGPU] extract_once failed for pair: "
                      << ts1 << " and " << ts2 << "\n";
            all_matches_[pair_count].clear();
            continue;
        }

        auto& c1 = cache[ts1];
        auto& c2 = cache[ts2];
        matcher.SetDescriptors(0, (int)c1.kpt.size(), c1.desc.data());
        matcher.SetDescriptors(1, (int)c2.kpt.size(), c2.desc.data());

        std::vector<std::pair<int,int>> matches;
        if (!c1.kpt.empty()) {
            std::unique_ptr<int[][2]> buf(new int[c1.kpt.size()][2]);
            int nmatch = matcher.GetSiftMatch((int)c1.kpt.size(), buf.get(), 0.7f, 0.8f, 1);
            matches.reserve(nmatch);
            for (int i = 0; i < nmatch; ++i) {
                int i1 = buf[i][0], i2 = buf[i][1];
                if (i1 >= 0 && i1 < (int)c1.kpt.size() &&
                    i2 >= 0 && i2 < (int)c2.kpt.size()) {
                    matches.emplace_back(i1, i2);
                }
            }
        }
        // std::cout << "matches.size(): " << matches.size() << std::endl;
        
        all_matches_[pair_count] = matches;

        // 可视化（沿用原先以“顺序下标”为文件名的函数签名）
        // 用 ts2idx 映射得到 int 下标，仍可复用 drawAndSaveMatchesGPU(int id1,int id2,...)
        int id1 = ts2idx[ts1];
        int id2 = ts2idx[ts2];
        cv::Mat img1 = cv::imread(getImagePath(ts1), cv::IMREAD_COLOR);
        cv::Mat img2 = cv::imread(getImagePath(ts2), cv::IMREAD_COLOR);
        drawAndSaveMatchesGPU(dataset_path_ + "result/", id1, id2, img1, img2, c1.kpt, c2.kpt, matches);
        pair_count++;
        // if (pair_count == 20) {
        //     std::cout << std::endl;
        //     std::cin.get(); // 仅示例，暂停查看前10个结果
        // }

        printProgressBar(pair_count, image_pairs_.size());
    }
    std::cout << std::endl;
}

void LvbaSystem::generateDepthWithVoxel() 
{
    const size_t N = all_voxel_ids_.size();
    if (poses_.size() != N) {
        std::cerr << "[generateDepthWithVoxel] size mismatch: poses=" << poses_.size()
                  << ", voxel_ids=" << N << std::endl;
    }
    if (!images_ids_.empty() && images_ids_.size() != N) {
        std::cerr << "[generateDepthWithVoxel] size mismatch: images_ids=" << images_ids_.size()
                  << ", voxel_ids=" << N << std::endl;
    }

    // std::cout << "all_voxel_ids_.size(): " << N << std::endl;

    Rcw_all_.clear(); tcw_all_.clear(); Rcw_all_optimized_.clear(); tcw_all_optimized_.clear(); all_depths_.clear();
    Rcw_all_.reserve(N); tcw_all_.reserve(N); Rcw_all_optimized_.reserve(N); tcw_all_optimized_.reserve(N); all_depths_.reserve(N);
    
    std::cout << "[generateDepthWithVoxel] Generating depths for " << N << " images ...\n";
    for (size_t id = 0; id < N; ++id) 
    {
        const Sophus::SE3& T_W_I_opt = poses_[id];
        const Eigen::Matrix3d Rwi_opt = T_W_I_opt.rotation_matrix();
        const Eigen::Vector3d Pwi_opt = T_W_I_opt.translation();

        Rcw_ = Rci_ * Rwi_opt.transpose();
        tcw_ = -Rcw_ * Pwi_opt + tci_;
        Rcw_all_optimized_.push_back(Rcw_);
        tcw_all_optimized_.push_back(tcw_);

        const Sophus::SE3& T_W_I_orig = poses_before_[id];
        const Eigen::Matrix3d Rwi_orig = T_W_I_orig.rotation_matrix();
        const Eigen::Vector3d Pwi_orig = T_W_I_orig.translation();
        Eigen::Matrix3d Rcw_orig = Rci_ * Rwi_orig.transpose();
        Eigen::Vector3d tcw_orig = -Rcw_orig * Pwi_orig + tci_;
        Rcw_all_.push_back(Rcw_orig);
        tcw_all_.push_back(tcw_orig);

        cv::Mat depth(image_height_, image_width_, CV_32FC1, cv::Scalar(0));

        const auto& voxel_ids = all_voxel_ids_[id];

        for (const VOXEL_LOC& voxel_xyz : voxel_ids) 
        {
            VOXEL_LOC position(voxel_xyz.x, voxel_xyz.y, voxel_xyz.z);
            auto it = grid_map_.find(position);
            if (it == grid_map_.end()) continue;

            const std::vector<Eigen::Vector3d>& points = it->second;

            for (const auto& pW : points)
            {
                Eigen::Vector3d pC = Rcw_ * pW + tcw_;
                
                const double Z = pC.z();
                if (Z < 1e-3) continue;

                const double x = pC.x() / Z;
                const double y = pC.y() / Z;

                const double r2 = x * x + y * y;
                const double x_dist = x * (1 + d0_ * r2 + d1_ * r2 * r2)
                                    + 2 * d2_ * x * y + d3_ * (r2 + 2 * x * x);
                const double y_dist = y * (1 + d0_ * r2 + d1_ * r2 * r2)
                                    + d2_ * (r2 + 2 * y * y) + 2 * d3_ * x * y;

                const int u = static_cast<int>(fx_ * x_dist + cx_);
                const int v = static_cast<int>(fy_ * y_dist + cy_);
                if (u < 0 || u >= image_width_ || v < 0 || v >= image_height_) continue;

                float& d = depth.at<float>(v, u);
                if (d == 0.f || Z < d) d = static_cast<float>(Z);
            }
        }

        all_depths_.push_back(depth);
        printProgressBar(all_depths_.size(), all_voxel_ids_.size());
        
        // 保存深度图，以时间戳命名
        if (!images_ids_.empty()) {
            std::ostringstream oss;
            oss.setf(std::ios::fixed); oss << std::setprecision(6) << images_ids_[id];
            const std::string out = dataset_path_ + "depth/" + oss.str() + ".png";
            cv::Mat vis;
            depth.convertTo(vis, CV_16UC1, 2000.0); // 1m -> 1000
            cv::imwrite(out, vis);
        }
    }
    std::cout << std::endl;

}

void LvbaSystem::BuildTracksAndFuse3D() {

    const int N = static_cast<int>(all_keypoints_.size());
    std::cout << "[BuildTracksAndFuse3D] Building visual points from " << N << " images ...\n";
    // 初始化 obs_to_track
    std::vector<std::vector<int>> obs_to_track(N);
    for (int i = 0; i < N; ++i) {
        obs_to_track[i].assign((int)all_keypoints_[i].size(), -1);
    }

    // 邻接表
    std::vector<std::vector<std::vector<std::pair<int,int>>>> adj(N);
    for (int i = 0; i < N; ++i) adj[i].resize(all_keypoints_[i].size());

    // 构建邻接表
    for (int i = 0; i < N-1; ++i) {
        for (int j = i+1; j < N; ++j) {
            size_t idx = pairIndex(i, j, N);
            const auto& matches_ij = all_matches_[idx];
            if (matches_ij.empty()) continue;
            for (const auto& m : matches_ij) {
                int ki = m.first;
                int kj = m.second;
                if (ki < 0 || kj < 0) continue;
                if (ki >= (int)all_keypoints_[i].size() || 
                    kj >= (int)all_keypoints_[j].size()) continue;
                adj[i][ki].push_back({j, kj});
                adj[j][kj].push_back({i, ki});
            }
        }
    }

    tracks_.clear();
    tracks_.reserve(100000);

    int num_tracked = 0;
    size_t total_components = 0;
    // BFS 建轨迹
    for (int i = 0; i < N; ++i) {
        for (int ki = 0; ki < (int)all_keypoints_[i].size(); ++ki) {
            if (obs_to_track[i][ki] != -1) continue;

            std::vector<std::pair<int,int>> component;
            std::deque<std::pair<int,int>> q;
            q.push_back({i, ki});
            obs_to_track[i][ki] = -2;

            while (!q.empty()) {
                auto cur = q.front(); q.pop_front();
                component.push_back(cur);
                int ci = cur.first;
                int ck = cur.second;
                for (auto& nb : adj[ci][ck]) {
                    int ni = nb.first, nk = nb.second;
                    if (obs_to_track[ni][nk] == -1) {
                        obs_to_track[ni][nk] = -2;
                        q.push_back(nb);
                    }
                }
            }
            ++total_components;

            if ((int)component.size() < obser_thr_) {
                for (auto &obs : component) obs_to_track[obs.first][obs.second] = -1;
                continue;
            }

            // 反投影所有观测
            std::vector<Eigen::Vector3d> points3d(component.size(), Eigen::Vector3d::Zero());
            std::vector<int> valid_mask(component.size(), 0);
            for (size_t t = 0; t < component.size(); ++t) {
                int im = component[t].first;
                int kp = component[t].second;
                float u = all_keypoints_[im][kp].x;
                float v = all_keypoints_[im][kp].y;

                float d = -1.0f;
                if (!fetchDepthBilinear(all_depths_[im], u, v, d, 0.001f)) continue;
                if (d <= 0.0f) continue;

                Eigen::Vector3d Xc = backProjectCam(u, v, d, fx_, fy_, cx_, cy_, d0_, d1_, d2_, d3_);
                Eigen::Vector3d Xw = camToWorld(Xc, Rcw_all_optimized_[im], tcw_all_optimized_[im]);
                points3d[t] = Xw;
                valid_mask[t] = 1;
            }

            std::vector<int> idx_valid;
            for (size_t t = 0; t < points3d.size(); ++t) if (valid_mask[t]) idx_valid.push_back((int)t);
            if ((int)idx_valid.size() < obser_thr_) {
                for (auto &obs : component) obs_to_track[obs.first][obs.second] = -1;
                continue;
            }

            Eigen::Vector3d anchor = points3d[idx_valid[0]];
            
            // 按距离挑 inliers（idx_valid 是 component 的下标子集）
            std::vector<int> inliers;
            inliers.reserve(idx_valid.size());
            for (int id : idx_valid) {
                double dist = (points3d[id] - anchor).norm();
                if (dist < 0.12) inliers.push_back(id);  // 0.1 m
            }
            if ((int)inliers.size() < obser_thr_) {
                for (auto &obs : component) obs_to_track[obs.first][obs.second] = -1;
                continue;
            }

            // 按 image_id 去重，计算融合点（每图一票）
            std::unordered_map<int,int> best_id;  // img_id -> chosen id
            best_id.reserve(inliers.size());
            for (int id : inliers) {
                int img_id = component[id].first;
                if (!best_id.count(img_id)) best_id[img_id] = id;
            }
            if ((int)best_id.size() < obser_thr_) {
                for (auto &obs : component) obs_to_track[obs.first][obs.second] = -1;
                continue;
            }

            // 融合并检测重投影误差
            Eigen::Vector3d Xw_fused = Eigen::Vector3d::Zero();
            for (const auto& kv : best_id)
            {
                const int comp_idx = kv.second;
                Xw_fused += points3d[comp_idx];
            }
            Xw_fused /= double(best_id.size());

            if (Xw_fused.norm() < 0.1) {
                std::cout << "bad fused point: " << Xw_fused.transpose() << std::endl;
            }

            // ---------- reprojection mean error gate (after Xw_fused) ----------
            double sum_err = 0.0;
            int cnt_err = 0;

            for (const auto& kv : best_id) {
                const int comp_idx = kv.second;
                const int img_id = component[comp_idx].first;
                const int kp_id  = component[comp_idx].second;

                if (img_id < 0 || img_id >= (int)Rcw_all_optimized_.size() || img_id >= (int)tcw_all_optimized_.size()) continue;
                if (img_id < 0 || img_id >= (int)all_keypoints_.size() || kp_id < 0 || kp_id >= (int)all_keypoints_[img_id].size()) continue;

                const double u_obs = all_keypoints_[img_id][kp_id].x;
                const double v_obs = all_keypoints_[img_id][kp_id].y;

                const Eigen::Matrix3d& Rcw = Rcw_all_optimized_[img_id];
                const Eigen::Vector3d& tcw = tcw_all_optimized_[img_id];
                const Eigen::Vector3d Xc = Rcw * Xw_fused + tcw;
                if (Xc.z() <= 1e-9) continue;

                const double invz = 1.0 / Xc.z();
                const double u_hat = fx_ * (Xc.x() * invz) + cx_;
                const double v_hat = fy_ * (Xc.y() * invz) + cy_;

                const double du = u_hat - u_obs;
                const double dv = v_hat - v_obs;
                const double err = std::sqrt(du * du + dv * dv);

                sum_err += err;
                cnt_err++;
            }

            if (cnt_err < obser_thr_) {
                for (auto &obs : component) obs_to_track[obs.first][obs.second] = -1;
                continue;
            }

            const double mean_reproj = sum_err / double(cnt_err);
            if (mean_reproj > reproj_mean_thr_px_) {

                std::cout << "[TrackFilter] drop by mean reproj=" << mean_reproj
                          << " thr=" << reproj_mean_thr_px_ << " cnt=" << cnt_err << std::endl;

                for (auto &obs : component) obs_to_track[obs.first][obs.second] = -1;
                continue;
            }

            const double cos_min_view_angle = std::cos(min_view_angle_deg_ * M_PI / 180.0);

            std::vector<int> kept_obs_ids;
            std::vector<Eigen::Vector3d> kept_dirs;
            kept_obs_ids.reserve(best_id.size());
            kept_dirs.reserve(best_id.size());

            for (auto &kv : best_id) {
                const int comp_idx = kv.second;
                const int cam_id = kv.first;
                if (cam_id < 0 || cam_id >= (int)Rcw_all_optimized_.size() ||
                    cam_id >= (int)tcw_all_optimized_.size()) {
                    continue;
                }
                const Eigen::Matrix3d& Rcw = Rcw_all_optimized_[cam_id];
                const Eigen::Vector3d& tcw = tcw_all_optimized_[cam_id];
                const Eigen::Vector3d Cw = -Rcw.transpose() * tcw;

                Eigen::Vector3d dir = points3d[comp_idx] - Cw;
                const double dir_norm = dir.norm();
                if (dir_norm < 1e-6) continue;
                dir /= dir_norm;

                double min_dot = 1.0;
                for (const auto& d : kept_dirs) {
                    const double dot = dir.dot(d);
                    if (dot < min_dot) min_dot = dot;
                }
                if (kept_dirs.empty() || min_dot <= cos_min_view_angle) {
                    kept_obs_ids.push_back(comp_idx);
                    kept_dirs.push_back(dir);
                }
            }

            // auto log_track_stats = [&](const char* tag) {
            //     static int dbg_cnt = 0;
            //     if (dbg_cnt < 10 || (dbg_cnt % 200 == 0)) {
            //         std::cout << "[TrackFilter] " << tag
            //                   << " comp=" << component.size()
            //                   << " inliers=" << inliers.size()
            //                   << " best=" << best_id.size()
            //                   << " kept=" << kept_obs_ids.size()
            //                   << std::endl;
            //     }
            //     ++dbg_cnt;
            // };

            if (kept_obs_ids.empty() || (int)kept_obs_ids.size() < obser_thr_) {
                // log_track_stats("drop");
                for (auto &obs : component) obs_to_track[obs.first][obs.second] = -1;
                continue;
            }
            // log_track_stats("keep");

            // Eigen::Vector3d Xw_fused = Eigen::Vector3d::Zero();
            // for (int id : kept_obs_ids) Xw_fused += points3d[id];
            // Xw_fused /= double(kept_obs_ids.size());

            // if (Xw_fused.norm() < 0.1) {
            //     std::cout << "bad fused point: " << Xw_fused.transpose() << std::endl;
            // }

            // RMS / MAD 基于去重后的集合
            // double sqsum = 0.0;
            // std::vector<double> resid_in;
            // resid_in.reserve(kept_obs_ids.size());
            // for (int id : kept_obs_ids) {
            //     double r = (points3d[id] - Xw_fused).norm();
            //     resid_in.push_back(r);
            //     sqsum += r * r;
            // }
            // double rms = std::sqrt(sqsum / double(resid_in.size()));
            // double mad_in = computeMAD(resid_in);

            // 写入 Track（inlier_indices 也是去重后的）
            Track tr;
            tr.Xw_fused = Xw_fused;
            // tr.mad = mad_in;
            // tr.rms = rms;
            tr.observations = component;
            tr.inlier_indices.reserve(kept_obs_ids.size());
            for (int id : kept_obs_ids) tr.inlier_indices.push_back(id);

            // 真正建成一个 track 再计数、再贴回 obs_to_track
            int track_id = (int)tracks_.size();
            tracks_.push_back(std::move(tr));
            ++num_tracked;

            for (auto &obs : component) obs_to_track[obs.first][obs.second] = track_id;
        }
    }
    if (total_components > 0) {
        const size_t kept = static_cast<size_t>(num_tracked);
        const size_t dropped = (total_components >= kept) ? (total_components - kept) : 0;
        const double keep_ratio = 100.0 * static_cast<double>(kept) / static_cast<double>(total_components);
        std::cout << "[TrackFilter] kept=" << kept << " dropped=" << dropped
                  << " total=" << total_components
                  << " ratio=" << std::fixed << std::setprecision(2)
                  << keep_ratio << "%" << std::defaultfloat << std::endl;
    }
    tracks_before_ = tracks_;

    all_depths_.clear();
    // showTracksComparePCL();
    // saveTrackFeaturesOnImages();
}


void LvbaSystem::buildGridMapFromOptimized() {
    grid_map_.clear();
    const auto& x_buf_full = dataset_io_->x_buf_;
    const auto& pl_fulls_full = dataset_io_->pl_fulls_;

    const size_t N = std::min(x_buf_full.size(), pl_fulls_full.size());
    if (N == 0) {
        ROS_WARN("buildGridMapFromOptimized: empty inputs, skip.");
        return;
    }

    const double vox = 0.5;

    size_t total_points = 0;
    std::vector<std::set<VOXEL_LOC>> per_frame_voxels(N);
    for (size_t i = 0; i < N; ++i) {
        const Eigen::Matrix3d& R = x_buf_full[i].R;
        const Eigen::Vector3d& t = x_buf_full[i].p;
        float loc_xyz[3];
        for (PointType& pc : pl_fulls_full[i]->points) {
            Eigen::Vector3d pvec_orig(pc.x, pc.y, pc.z);
            Eigen::Vector3d pvec_tran = R * pvec_orig + t;
            for (int j = 0; j < 3; ++j) {
                loc_xyz[j] = pvec_tran[j] / vox;
                if (loc_xyz[j] < 0.0f) loc_xyz[j] -= 1.0f;
            }
            VOXEL_LOC position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]);
            grid_map_[position].push_back(pvec_tran);
            per_frame_voxels[i].insert(VOXEL_LOC{(int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]});
            ++total_points;
        }
    }


    const double half_w = 0.5;
    std::vector<double> pcd_ts;
    pcd_ts.reserve(x_buf_full.size());
    for (const auto& kv : x_buf_full) pcd_ts.push_back(kv.t);

    all_voxel_ids_.clear();
    all_voxel_ids_.reserve(images_ids_.size());

    for (const double& img_id : images_ids_) {
        double t_img = 0.0;
        std::string img_name_str = std::to_string(img_id);
        if (!parseTimestampFromName(img_name_str, t_img)) {
            std::cerr << "[buildGridMap] bad image id in images_ids_: " << img_name_str << "\n";
            all_voxel_ids_.emplace_back();
            continue;
        }

        const double t0 = t_img - half_w;
        const double t1 = t_img + half_w;
        auto itL = std::lower_bound(pcd_ts.begin(), pcd_ts.end(), t0);
        auto itR = std::upper_bound(pcd_ts.begin(), pcd_ts.end(), t1);

        std::set<VOXEL_LOC> voxels_set;
        for (auto it = itL; it != itR; ++it) {
            const size_t idx = static_cast<size_t>(it - pcd_ts.begin());
            if (idx >= per_frame_voxels.size()) continue;
            voxels_set.insert(per_frame_voxels[idx].begin(), per_frame_voxels[idx].end());
        }

        std::vector<VOXEL_LOC> one_vox;
        one_vox.reserve(voxels_set.size());
        for (const auto& v : voxels_set) one_vox.push_back(v);
        all_voxel_ids_.push_back(std::move(one_vox));
    }
    std::cout << "[buildGridMap] built global world cloud points=" << total_points
              << " from pcds=" << pcd_ts.size() << "\n";
    std::cout << "[buildGridMap] voxel ids: images=" << images_ids_.size()
              << ", merged window=±" << half_w << "s, vox_size=" << vox << "\n";
}

void LvbaSystem::saveTrackFeaturesOnImages()
{
    if (images_ids_.empty() || all_keypoints_.empty()) {
        std::cerr << "[TrackFeature] skip: empty images/keypoints.\n";
        return;
    }
    if (all_keypoints_.size() != images_ids_.size()) {
        std::cerr << "[TrackFeature] size mismatch: keypoints=" << all_keypoints_.size()
                  << " images=" << images_ids_.size() << "\n";
        return;
    }

    std::vector<std::vector<char>> used(all_keypoints_.size());
    for (size_t i = 0; i < all_keypoints_.size(); ++i) {
        used[i].assign(all_keypoints_[i].size(), 0);
    }

    for (const auto& tr : tracks_) {
        for (int idx_in_obs : tr.inlier_indices) {
            if (idx_in_obs < 0 || idx_in_obs >= (int)tr.observations.size()) continue;
            const auto& obs = tr.observations[idx_in_obs];
            const int cam_id = obs.first;
            const int kp_id = obs.second;
            if (cam_id < 0 || cam_id >= (int)used.size()) continue;
            if (kp_id < 0 || kp_id >= (int)used[cam_id].size()) continue;
            used[cam_id][kp_id] = 1;
        }
    }

    const std::string out_dir = dataset_path_ + "track_features/";
    if (!fs::exists(out_dir)) fs::create_directories(out_dir);

    size_t total_drawn = 0;
    size_t total_sift = 0;
    for (size_t i = 0; i < images_ids_.size(); ++i) {
        const double img_id = images_ids_[i];
        cv::Mat img = cv::imread(getImagePath(img_id), cv::IMREAD_COLOR);
        if (img.empty()) {
            std::cerr << "[TrackFeature] cannot read image: " << img_id << "\n";
            continue;
        }

        if (img.cols != image_width_ || img.rows != image_height_) {
            cv::resize(img, img, cv::Size(img.cols * scale_, img.rows * scale_), 0, 0, cv::INTER_LINEAR);
        }

        const auto& kpts = all_keypoints_[i];
        for (const auto& kp : kpts) {
            cv::circle(img, cv::Point2f(kp.x, kp.y), 2, CV_RGB(255,0,0), -1, cv::LINE_AA);
        }

        size_t count = 0;
        for (size_t k = 0; k < used[i].size(); ++k) {
            if (!used[i][k]) continue;
            const auto& kp = all_keypoints_[i][k];
            cv::circle(img, cv::Point2f(kp.x, kp.y), 2, CV_RGB(0,255,0), -1, cv::LINE_AA);
            ++count;
        }
        total_drawn += count;
        total_sift += kpts.size();

        const std::string text = "sift=" + std::to_string(kpts.size()) + " track=" + std::to_string(count);
        cv::putText(img, text, cv::Point(12, 24), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                    CV_RGB(255,255,255), 2, cv::LINE_AA);
        cv::putText(img, text, cv::Point(12, 24), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                    CV_RGB(0,0,0), 1, cv::LINE_AA);

        const std::string out_path = out_dir + std::to_string(img_id) + ".png";
        if (!cv::imwrite(out_path, img)) {
            std::cerr << "[TrackFeature] failed to write: " << out_path << "\n";
        }

        std::cout << "[TrackFeature] img_id=" << img_id
                  << " sift=" << kpts.size()
                  << " track=" << count << "\n";
    }

    std::cout << "[TrackFeature] saved images to " << out_dir
              << " total_sift=" << total_sift
              << " total_track=" << total_drawn << "\n";
}


void LvbaSystem::optimizeCameraPoses()
{
    // ---------------- 基本检查 ----------------
    const int M = static_cast<int>(Rcw_all_.size());
    if (M == 0 || (int)tcw_all_.size() != M)
        throw std::runtime_error("optimizeCamPoses: Rcw_all_/tcw_all_ size mismatch or empty.");
    if ((int)all_keypoints_.size() != M)
        throw std::runtime_error("optimizeCamPoses: all_keypoints_.size() must equal #cameras.");
    if (tracks_.empty())
        throw std::runtime_error("optimizeCamPoses: tracks_ is empty. Run BuildTracksAndFuse3D() first.");

    std::vector<int> track_ids; track_ids.reserve(tracks_.size());
    for (int i = 0; i < (int)tracks_.size(); ++i) {
        const auto& tr = tracks_[i];
        if (tr.observations.size() >= obser_thr_ && !tr.Xw_fused.isZero(1e-12) && tr.Xw_fused.allFinite()) {
            track_ids.push_back(i);
        }
    }
    
    if (track_ids.empty()) {
        std::cerr << "[optimizeCamPoses] Warning: no usable tracks found!" << std::endl;
        return;
    }

    const int Npts = (int)track_ids.size();
    std::cout << "[optimizeCamPoses] usable tracks = " << Npts << ", cameras = " << M << std::endl;

    const double surf_voxel_size = dataset_io_->stage2_root_voxel_size_;
    const float surf_eigen_thr = dataset_io_->stage2_eigen_ratio_array_[0];

    const auto& pl_fulls = dataset_io_->pl_fulls_;
    const auto& x_buf_full = dataset_io_->x_buf_;
    const int total_size = static_cast<int>(std::min(pl_fulls.size(), x_buf_full.size()));
    if (total_size == 0) {
        std::cerr << "[optimizeCamPoses] empty pl_fulls/x_buf, skip." << std::endl;
        return;
    }

    std::vector<IMUST> anchor_poses;
    std::vector<pcl::PointCloud<PointType>::Ptr> anchor_clouds;

    const int window_size = dataset_io_->window_ba_size_;
    const double anchor_leaf = dataset_io_->anchor_leaf_size_;

    anchor_poses.reserve((total_size + window_size - 1) / window_size);
    anchor_clouds.reserve((total_size + window_size - 1) / window_size);

    for (int start = 0; start < total_size; start += window_size) {
        const int end = std::min(start + window_size, total_size);
        const int curr_win = end - start;
        if (curr_win <= 0) break;

        pcl::PointCloud<PointType>::Ptr merged(new pcl::PointCloud<PointType>());
        const IMUST anchor_pose = x_buf_full[start];

        for (int j = start; j < end; ++j) {
            pcl::PointCloud<PointType> tmp = *pl_fulls[j];
            IMUST rel;
            rel.R = anchor_pose.R.transpose() * x_buf_full[j].R;
            rel.p = anchor_pose.R.transpose() * (x_buf_full[j].p - anchor_pose.p);
            pl_transform(tmp, rel);
            *merged += tmp;
        }

        down_sampling_voxel2(*merged, anchor_leaf);
        anchor_poses.push_back(anchor_pose);
        anchor_clouds.push_back(merged);
    }

    const int anchor_size = static_cast<int>(std::min(anchor_poses.size(), anchor_clouds.size()));
    if (anchor_size == 0) {
        std::cerr << "[optimizeCamPoses] empty anchor_poses/anchor_clouds, skip." << std::endl;
        return;
    }

    std::unordered_map<VOXEL_LOC, OCTO_TREE_ROOT*> surf_map;
    for (int j = 0; j < anchor_size; ++j) {
        cut_voxel(surf_map, *anchor_clouds[j], anchor_poses[j], j, anchor_size,
                  surf_voxel_size, surf_eigen_thr);
    }
    printf("surf_map.size(): %zu\n", surf_map.size());
    for (auto& kv : surf_map) {
        if (kv.second != nullptr) kv.second->recut(anchor_poses);
    }
    printf("After recut, surf_map.size(): %zu\n", surf_map.size());

    // ---------------- 初始化优化变量 ----------------
    std::vector<std::array<double,4>> qs(M);
    std::vector<std::array<double,3>> ts(M);
    
    // 初始化 Pose
    for (int k = 0; k < M; ++k) {
        Eigen::Quaterniond q_eig(Rcw_all_optimized_[k]); q_eig.normalize();
        qs[k] = { q_eig.w(), q_eig.x(), q_eig.y(), q_eig.z() };
        ts[k] = { tcw_all_optimized_[k].x(), tcw_all_optimized_[k].y(), tcw_all_optimized_[k].z() };
    }
    
    // 初始化 Points
    std::vector<std::array<double,3>> Xs(Npts);
    for (int pi = 0; pi < Npts; ++pi) {
        const auto& X = tracks_[ track_ids[pi] ].Xw_fused;
        Xs[pi] = { X.x(), X.y(), X.z() };
    }

    // 平面缓存
    std::vector<Eigen::Vector3d> plane_n(Npts, Eigen::Vector3d::Zero());
    std::vector<double>          plane_d(Npts, 0.0);

    auto recompute_local_planes = [&](){
        for (int pi = 0; pi < Npts; ++pi)
        {
            Eigen::Vector3d X(Xs[pi][0], Xs[pi][1], Xs[pi][2]);
            if (!X.allFinite()) { // 检查点是否有效
                plane_n[pi].setZero(); plane_d[pi] = 0.0; continue;
            }

            float loc_xyz[3];
            for (int j = 0; j < 3; ++j) {
                loc_xyz[j] = X[j] / surf_voxel_size;
                if (loc_xyz[j] < 0) loc_xyz[j] -= 1.0f;
            }
            VOXEL_LOC key((int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]);
            
            auto it = surf_map.find(key);
            if (it == surf_map.end() || it->second == nullptr) {
                plane_n[pi].setZero(); plane_d[pi] = 0.0; continue;
            }

            OCTO_TREE_NODE* node = it->second->findCorrespondPoint(X);
            if (node == nullptr || node->octo_state != PLANE) {
                plane_n[pi].setZero(); plane_d[pi] = 0.0; continue;
            }

            if (!node->direct.allFinite() || node->direct.norm() < 1e-6 || !node->center.allFinite()) {
                plane_n[pi].setZero(); plane_d[pi] = 0.0; continue;
            }

            Eigen::Vector3d n = node->direct;
            n.normalize();
            plane_n[pi] = n;
            plane_d[pi] = -n.dot(node->center);
        }
    };

    // 计算平面
    recompute_local_planes();
    for (auto& kv : surf_map) delete kv.second;

    ceres::Problem problem;
    ceres::Solver::Options options;
    options.max_num_iterations = 50; 
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.num_threads = std::max(1u, std::thread::hardware_concurrency());
    options.minimizer_progress_to_stdout = true;

    for (int k = 0; k < M; ++k) {
        problem.AddParameterBlock(qs[k].data(), 4, new ceres::EigenQuaternionManifold());
        problem.AddParameterBlock(ts[k].data(), 3);
    }
    problem.SetParameterBlockConstant(qs[0].data());
    problem.SetParameterBlockConstant(ts[0].data());

    ceres::LossFunction* loss_function_reproj = new ceres::HuberLoss(1.0); 
    ceres::LossFunction* loss_function_plane  = new ceres::HuberLoss(0.1); 

    std::vector<bool> point_is_valid(Npts, false);

    const double sigma_px = 0.5;
    const double sigma_plane = 0.01;

    for (int pi = 0; pi < Npts; ++pi) {
        
        const Eigen::Vector3d& n = plane_n[pi];
        const double d = plane_d[pi];

        bool has_valid_plane = (n.allFinite() && std::isfinite(d) && !n.isZero(1e-6));
        
        if (!has_valid_plane) {
            point_is_valid[pi] = false; 
            continue; 
        }

        // 只有通过了上面的筛选，才标记为有效
        point_is_valid[pi] = true;

        // 添加 Point 参数块 (因为有平面，所以添加)
        problem.AddParameterBlock(Xs[pi].data(), 3);

        // 添加 视觉重投影残差
        const int tid = track_ids[pi];
        const auto& tr = tracks_[tid]; 
        std::unordered_set<int> seen;
        for (int idx_in_obs : tr.inlier_indices) {
            if (idx_in_obs < 0 || idx_in_obs >= (int)tr.observations.size()) continue;
            if (!seen.insert(idx_in_obs).second) continue;
    
            const auto& obs = tr.observations[idx_in_obs];
            const int cam_id = obs.first;
            const int kp_id  = obs.second;
            if (cam_id < 0 || cam_id >= M) continue;
    
            const double u = all_keypoints_[cam_id][kp_id].x;
            const double v = all_keypoints_[cam_id][kp_id].y;

            ceres::CostFunction* cost = ReprojErrorWhitenedDistorted::Create(
                    u, v, fx_, fy_, cx_, cy_, d0_, d1_, d2_, d3_, sigma_px, sigma_px);
            
            problem.AddResidualBlock(cost, nullptr,
                                     qs[cam_id].data(), ts[cam_id].data(), Xs[pi].data());
        }

        // 添加 点-面残差
        // double r10 = (std::abs(n(0)) < 1e-12) ? 1e12 : std::abs(n(1)/n(0));
        // double r12 = (std::abs(n(2)) < 1e-12) ? 1e12 : std::abs(n(1)/n(2));
        // sigma_plane = (r10>10 && r12>10) ? 0.02 : 0.05; 
        ceres::CostFunction* plane_cost = PointPlaneErrorWhitened::Create(n, d, sigma_plane);
        problem.AddResidualBlock(plane_cost, nullptr, Xs[pi].data());
    }

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << "[optimizeCamPoses] " << summary.BriefReport() << std::endl;

    if (summary.termination_type == ceres::FAILURE) {
        std::cerr << "[optimizeCamPoses] Solver failed!" << std::endl;
        return;
    }

    for (int k = 0; k < M; ++k) {
        Eigen::Quaterniond q_eig(qs[k][0], qs[k][1], qs[k][2], qs[k][3]);
        q_eig.normalize();
        Rcw_all_optimized_[k] = q_eig.toRotationMatrix();
        tcw_all_optimized_[k] = Eigen::Vector3d(ts[k][0], ts[k][1], ts[k][2]);
    }

    int valid_cnt = 0;
    for (int pi = 0; pi < Npts; ++pi) {
        if (point_is_valid[pi]) {
            Eigen::Vector3d X_new(Xs[pi][0], Xs[pi][1], Xs[pi][2]);
            tracks_[ track_ids[pi] ].Xw_fused = X_new;
            valid_cnt++;
        } 
    }
    
    std::cout << "[optimizeCamPoses] Points kept: " << valid_cnt << " / " << Npts << std::endl;

    std::cout << "[optimizeCamPoses] done." << std::endl;
}

void LvbaSystem::visualizeProj() {

    namespace fs = std::filesystem;

    // ------- 小工具（lambda） -------
    auto drawCross = [](cv::Mat& img, const cv::Point2d& p, int size, int thickness, const cv::Scalar& color) {
        cv::line(img, cv::Point2d(p.x - size, p.y), cv::Point2d(p.x + size, p.y), color, thickness, cv::LINE_AA);
        cv::line(img, cv::Point2d(p.x, p.y - size), cv::Point2d(p.x, p.y + size), color, thickness, cv::LINE_AA);
    };
    auto putTextShadow = [](cv::Mat& img, const std::string& text, cv::Point org, double scale=0.45, int thick=1, cv::Scalar color=CV_RGB(0,0,0)) {
        // cv::putText(img, text, org + cv::Point(1,1), cv::FONT_HERSHEY_SIMPLEX, scale, CV_RGB(0,0,0), thick+2, cv::LINE_AA);
        cv::putText(img, text, org, cv::FONT_HERSHEY_SIMPLEX, scale, color, thick, cv::LINE_AA);
    };
    auto projectWithDistortion = [&](const Eigen::Vector3d& Pw,
                                     const Eigen::Matrix3d& Rcw,
                                     const Eigen::Vector3d& tcw,
                                     cv::Point2d& uv_out) -> bool {
        const Eigen::Vector3d Pc = Rcw * Pw + tcw;
        const double X = Pc.x(), Y = Pc.y(), Z = Pc.z();
        if (!(Z > 1e-12) || !std::isfinite(X) || !std::isfinite(Y) || !std::isfinite(Z)) return false;
                                
        const double x = X / Z, y = Y / Z;
        const double r2 = x*x + y*y, r4 = r2*r2;
        const double radial = 1.0 + d0_ * r2 + d1_ * r4;
        const double x_t = 2.0*d2_*x*y + d3_*(r2 + 2.0*x*x);
        const double y_t = d2_*(r2 + 2.0*y*y) + 2.0*d3_*x*y;
        const double xd = x*radial + x_t;
        const double yd = y*radial + y_t;

        const double u = fx_ * xd + cx_;
        const double v = fy_ * yd + cy_;
        if (!std::isfinite(u) || !std::isfinite(v)) return false;
        uv_out = cv::Point2d(u, v);
        return true;
    };

    // ------- 基本检查 -------
    const int M = static_cast<int>(images_ids_.size());
    if (M == 0) {
        std::cerr << "[visualizeProj] images_ids_ is empty.\n";
        return;
    }
    if (Rcw_all_.size() != (size_t)M || tcw_all_.size() != (size_t)M ||
        Rcw_all_optimized_.size() != (size_t)M || tcw_all_optimized_.size() != (size_t)M) {
        std::cerr << "[visualizeProj] pose arrays size mismatch with images_ids_.\n";
        return;
    }

    // 输出目录
    std::string out_dir = dataset_path_ + "reproj";
    if (!fs::exists(out_dir)) fs::create_directories(out_dir);

    // 是否有优化后三维点
    bool has_after_pts = false;
    if (this->tracks_.size() == this->tracks_before_.size() && !this->tracks_.empty()) {
        has_after_pts = true;
    }

    // 按图像聚合条目
    struct Item {
        cv::Point2d uv_meas;
        bool has_pre=false, has_post=false;
        cv::Point2d uv_pre, uv_post;
        double err_pre=-1, err_post=-1;
        int track_id=-1;
    };
    std::vector<std::vector<Item>> per_image_items(M);

    // -------- 收集：仅使用内点观测 --------
    for (int tid = 0; tid < (int)tracks_before_.size(); ++tid) {
        const auto& tr_b = tracks_before_[tid];
        const auto& tr_a = tracks_[tid]; // 优化后


        const Eigen::Vector3d Pw_pre  = tr_b.Xw_fused;
        const Eigen::Vector3d Pw_post = tr_a.Xw_fused; // 若没有优化后三维点，则等于 pre

        std::unordered_set<int> seen;
        seen.reserve(tr_b.inlier_indices.size());

        for (int idx_in_obs : tr_b.inlier_indices) {
            if (idx_in_obs < 0 || idx_in_obs >= (int)tr_b.observations.size()) continue;
            if (!seen.insert(idx_in_obs).second) continue;

            const auto& obs = tr_b.observations[idx_in_obs];
            const int cam_id = obs.first;
            const int kp_id  = obs.second;
            if (cam_id < 0 || cam_id >= M) continue;
            if (kp_id  < 0 || kp_id >= (int)all_keypoints_[cam_id].size()) continue;

            const double u_meas = all_keypoints_[cam_id][kp_id].x;
            const double v_meas = all_keypoints_[cam_id][kp_id].y;
            cv::Point2d uv_meas(u_meas, v_meas);

            // pre
            cv::Point2d uv_pre;
            bool ok_pre = projectWithDistortion(Pw_pre, Rcw_all_[cam_id], tcw_all_[cam_id], uv_pre);
            // post
            cv::Point2d uv_post;
            bool ok_post = projectWithDistortion(Pw_post, Rcw_all_optimized_[cam_id], tcw_all_optimized_[cam_id], uv_post);

            Item it;
            it.uv_meas = uv_meas;
            it.has_pre = ok_pre;
            it.has_post = ok_post;
            it.uv_pre = uv_pre;
            it.uv_post = uv_post;
            it.err_pre  = ok_pre  ? cv::norm(uv_pre  - uv_meas) : -1.0;
            it.err_post = ok_post ? cv::norm(uv_post - uv_meas) : -1.0;
            it.track_id = tid;

            per_image_items[cam_id].push_back(std::move(it));
        }
    }

    // -------- 绘制并保存 --------
    double global_err_pre = 0.0, global_err_post = 0.0;
    int global_cnt = 0;
    for (int k = 0; k < M; ++k) {
        const double img_id = images_ids_[k];
        const std::string img_path = getImagePath(img_id);
        cv::Mat img = cv::imread(img_path, cv::IMREAD_COLOR);
        if (img.empty()) {
            std::cerr << "[visualizeProj] cannot read image: " << img_path << "\n";
            continue;
        }

        double sum_pre = 0.0, sum_post = 0.0;
        int cnt_pre = 0, cnt_post = 0;

        for (const auto& it : per_image_items[k]) {
            // 测量点：绿十字
            drawCross(img, it.uv_meas, 5, 1, CV_RGB(0,255,0));

            // pre：蓝点 + 线
            if (it.has_pre) {
                cv::circle(img, it.uv_pre, 2, CV_RGB(0,128,255), -1, cv::LINE_AA);
                // cv::line(img, it.uv_pre, it.uv_meas, CV_RGB(0,128,255), 1, cv::LINE_AA);
                sum_pre += it.err_pre; cnt_pre++;
            }
            // post：红点 + 线
            if (it.has_post) {
                cv::rectangle(img, cv::Point(it.uv_post.x-1, it.uv_post.y-1), cv::Point(it.uv_post.x+1, it.uv_post.y+1), CV_RGB(255,0,0), -1, cv::LINE_AA);
                // cv::line(img, it.uv_post, it.uv_meas, CV_RGB(255,0,0), 1, cv::LINE_AA);
                sum_post += it.err_post; cnt_post++;
            }
        }

        const double mean_pre  = (cnt_pre  > 0) ? (sum_pre  / cnt_pre)  : -1.0;
        const double mean_post = (cnt_post > 0) ? (sum_post / cnt_post) : -1.0;
        global_cnt++; 
        global_err_pre += mean_pre; 
        global_err_post += mean_post;
        // 角标信息 + 图例
        {
            std::ostringstream head;
            head.setf(std::ios::fixed); head.precision(3);
            head << "img_id=" << img_id
                 << "  N=" << per_image_items[k].size()
                 << "  mean_pre=" << mean_pre
                 << "  mean_post=" << mean_post;
            putTextShadow(img, head.str(), cv::Point(12, 24), 0.9 * scale_, 1, CV_RGB(0,0,0));
            putTextShadow(img, "meas: green cross",      cv::Point(12, 48), 0.55 * scale_, 1, CV_RGB(0,255,0));
            putTextShadow(img, "pre:  blue dot",  cv::Point(12, 68), 0.55 * scale_, 1, CV_RGB(0,0,255));
            putTextShadow(img, "post: red rectangle", cv::Point(12, 88), 0.55 * scale_, 1, CV_RGB(255,0,0));
        }

        char out_name[512];
        std::snprintf(out_name, sizeof(out_name), "%s/vis_%08.0f.png", out_dir.c_str(), img_id);
        if (!cv::imwrite(out_name, img)) {
            std::cerr << "[visualizeProj] failed to write: " << out_name << "\n";
        }
    }
    global_err_pre /= global_cnt;
    global_err_post /= global_cnt;
    std::cout << "[visualizeProj] global mean pre: " << global_err_pre << "\n";
    std::cout << "[visualizeProj] global mean post: " << global_err_post << "\n";

    std::cout << "[visualizeProj] done. saved to: " << out_dir << std::endl;
    std::string dir = dataset_path_ + "Colmap/colored_merged.pcd";
    VisualizeOptComparison(images_ids_, true, dir);
}

void LvbaSystem::showTracksComparePCL() 
{
    std::cout << "[Visualizer] Preparing data..." << std::endl;

    pcl::PointCloud<pcl::PointXYZ>::Ptr track_viz_before(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr track_viz_after (new pcl::PointCloud<pcl::PointXYZ>());
    
    track_viz_before->reserve(tracks_before_.size());
    for (const auto& tr : tracks_before_) {
        const auto& X = tr.Xw_fused;
        if (X.allFinite()) track_viz_before->emplace_back((float)X.x(), (float)X.y(), (float)X.z());
    }
    
    track_viz_after->reserve(tracks_.size());
    for (const auto& tr : tracks_) {
        const auto& X = tr.Xw_fused;
        if (X.allFinite()) track_viz_after->emplace_back((float)X.x(), (float)X.y(), (float)X.z());
    }

    std::cout << "[Visualizer] Before: " << track_viz_before->size() << " | After: " << track_viz_after->size() << std::endl;

    std::string target_frame_id = "map"; 
    ros::Time current_time = ros::Time::now();
    
    if (track_viz_before->size() > 0) {
        sensor_msgs::PointCloud2 msg_before;
        pcl::toROSMsg(*track_viz_before, msg_before);
        msg_before.header.frame_id = target_frame_id;
        msg_before.header.stamp = current_time;
        pub_cloud_before_.publish(msg_before);
    }

    if (track_viz_after->size() > 0) {
        sensor_msgs::PointCloud2 msg_after;
        pcl::toROSMsg(*track_viz_after, msg_after);
        msg_after.header.frame_id = target_frame_id;
        msg_after.header.stamp = current_time;
        pub_cloud_after_.publish(msg_after);
    }
}

void LvbaSystem::drawAndSaveMatchesGPU(
    const std::string& out_dir,
    int id1, int id2,
    const cv::Mat& img1, const cv::Mat& img2,
    const std::vector<SiftGPU::SiftKeypoint>& kpts1,
    const std::vector<SiftGPU::SiftKeypoint>& kpts2,
    const std::vector<std::pair<int,int>>& matches) {

    namespace fs = std::filesystem;
    fs::create_directories(out_dir);

    // 拼接画布
    int H = std::max(img1.rows, img2.rows);
    int W = img1.cols + img2.cols;
    cv::Mat canvas(H, W, CV_8UC3, cv::Scalar(20,20,20));
    img1.copyTo(canvas(cv::Rect(0,0,img1.cols,img1.rows)));
    img2.copyTo(canvas(cv::Rect(img1.cols,0,img2.cols,img2.rows)));

    // 随机颜色
    cv::RNG rng(12345);
    auto randColor = [&](){ return cv::Scalar(rng.uniform(64,255),
                                                rng.uniform(64,255),
                                                rng.uniform(64,255)); };

    for (auto& m : matches) {
        int i1 = m.first, i2 = m.second;
        if (i1<0 || i1>=(int)kpts1.size() || i2<0 || i2>=(int)kpts2.size()) continue;

        cv::Point2f p1(kpts1[i1].x, kpts1[i1].y);
        cv::Point2f p2(kpts2[i2].x + img1.cols, kpts2[i2].y);

        auto col = randColor();
        cv::circle(canvas, p1, 3, col, -1, cv::LINE_AA);
        cv::circle(canvas, p2, 3, col, -1, cv::LINE_AA);
        cv::line(canvas, p1, p2, col, 1, cv::LINE_AA);
    }
    std::cout << " Drawed : " << id1 << " - " << id2
              << " | matches: " << matches.size() << std::endl;
    std::string save_path = out_dir + "/" + std::to_string(id1+1) + "_" + std::to_string(id2+1) + "_matches_nums:" + std::to_string(matches.size())+".jpg";
    cv::imwrite(save_path, canvas);
}

bool LvbaSystem::ProjectToImage(
    const Eigen::Matrix3d& Rcw, const Eigen::Vector3d& tcw,
    const Eigen::Vector3d& Xw,
    double* u, double* v, double* Zc) const
{
    const Eigen::Vector3d Xc = Rcw * Xw + tcw;
    const double z = Xc.z();
    if (Zc) *Zc = z;
    if (z <= 1e-6) return false;

    const double xn = Xc.x() / z;
    const double yn = Xc.y() / z;

    // Brown-Conrady: k1,k2,p1,p2
    const double k1 = d0_, k2 = d1_, p1 = d2_, p2 = d3_;
    const double r2  = xn*xn + yn*yn;
    const double r4  = r2 * r2;
    const double radial = 1.0 + k1 * r2 + k2 * r4;

    // 切向
    const double x_tan = 2.0 * p1 * xn * yn + p2 * (r2 + 2.0 * xn * xn);
    const double y_tan = p1 * (r2 + 2.0 * yn * yn) + 2.0 * p2 * xn * yn;

    // 畸变后归一化坐标
    const double xdist = xn * radial + x_tan;
    const double ydist = yn * radial + y_tan;

    // 像素坐标
    const double uu = fx_ * xdist + cx_;
    const double vv = fy_ * ydist + cy_;

    if (u) *u = uu;
    if (v) *v = vv;
    return std::isfinite(uu) && std::isfinite(vv);
}


void LvbaSystem::VisualizeOptComparison(
    const std::vector<double>& image_ids,
    bool save_merged_pcd,
    const std::string& merged_pcd_path)
{
    const auto& pl_fulls = dataset_io_->pl_fulls_;
    const auto& x_buf_opt = dataset_io_->x_buf_;
    const auto& x_buf_bef = dataset_io_->x_buf_before_;

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr merged(new pcl::PointCloud<pcl::PointXYZRGB>());
    merged->reserve(3000000);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr merged_b(new pcl::PointCloud<pcl::PointXYZRGB>());
    merged_b->reserve(3000000);

    if(colmap_output_enable_)
    {
        std::string sparse_dir = dataset_path_ + "Colmap/sparse/";
        if (!fs::exists(sparse_dir)) fs::create_directories(sparse_dir);
        fout_poses_after.open(sparse_dir + "images.txt", std::ios::out);
        // fout_poses_before.open(dataset_path_ + "Colmap/before_sparse/images.txt", std::ios::out);
    }
    
    for (size_t k = 0; k < image_ids.size(); ++k) {
        const double img_id = image_ids[k];
        const std::string img_path = getImagePath(img_id);        

        // 读图（BGR）
        cv::Mat img = cv::imread(img_path, cv::IMREAD_COLOR);
        if (img.empty()) {
            std::cerr << "[Colorize] Failed to load image: " << img_path << "\n";
            continue;
        }
        if (img.cols != image_width_ || img.rows != image_height_) 
        {
            cv::resize(img, img, cv::Size(img.cols * scale_, img.rows * scale_), 0, 0, cv::INTER_LINEAR);        
        }
        const int W = img.cols, H = img.rows;

        //-----------------------多帧合并（内存点云）----------------------//
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_w_all_opt(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_w_all_orig(new pcl::PointCloud<pcl::PointXYZ>());

        for (size_t idx = 0; idx < x_buf_opt.size(); ++idx) {
            if (std::fabs(x_buf_opt[idx].t - img_id) > 0.5) {
                continue;
            }
            if (idx >= pl_fulls.size()) continue;
            const auto& pl_body = pl_fulls[idx];
            const IMUST& pose_opt = x_buf_opt[idx];
            const IMUST& pose_bef = x_buf_bef[idx];

            for (const auto& pb : pl_body->points) {
                Eigen::Vector3d Xw_opt = pose_opt.R * Eigen::Vector3d(pb.x, pb.y, pb.z) + pose_opt.p;
                cloud_w_all_opt->emplace_back(static_cast<float>(Xw_opt.x()),
                                              static_cast<float>(Xw_opt.y()),
                                              static_cast<float>(Xw_opt.z()));
                Eigen::Vector3d Xw_orig = pose_bef.R * Eigen::Vector3d(pb.x, pb.y, pb.z) + pose_bef.p;
                cloud_w_all_orig->emplace_back(static_cast<float>(Xw_orig.x()),
                                               static_cast<float>(Xw_orig.y()),
                                               static_cast<float>(Xw_orig.z()));
            }
        }
        if (cloud_w_all_opt->empty() || cloud_w_all_orig->empty()) {
            std::cerr << "[Colorize] skip image " << img_id << " no lidar in window\n";
            continue;
        }

        const Eigen::Matrix3d& Rcw = Rcw_all_optimized_[k];
        const Eigen::Vector3d& tcw = tcw_all_optimized_[k];
        const Eigen::Matrix3d& Rcw_b = Rcw_all_[k];
        const Eigen::Vector3d& tcw_b = tcw_all_[k];
        
        // Colmap 格式
        Eigen::Quaterniond q(Rcw);
        Eigen::Vector3d t = tcw;
        Eigen::Quaterniond q_b(Rcw_b);
        Eigen::Vector3d t_b = tcw_b;
        
        if(colmap_output_enable_)
        {        
            // fout_poses_before << k << " "
            //        << std::fixed << std::setprecision(6)
            //        << q_b.w() << " " << q_b.x() << " " << q_b.y() << " " << q_b.z() << " "
            //        << t_b.x() << " " << t_b.y() << " " << t_b.z() << " "
            //        << 1 << " "  // CAMERA_ID
            //        << k << ".jpg" << std::endl;
            // fout_poses_before << "0.0 0.0 -1" << std::endl;
            fout_poses_after << k << " "
                    << std::fixed << std::setprecision(6)
                    << q.w() << " " << q.x() << " " << q.y() << " " << q.z() << " "
                    << t.x() << " " << t.y() << " " << t.z() << " "
                    << 1 << " "  // CAMERA_ID
                    << k << ".jpg" << std::endl;
            fout_poses_after << "0.0 0.0 -1" << std::endl;

            std::string images_dir = dataset_path_ + "Colmap/images/";
            if (!fs::exists(images_dir)) fs::create_directories(images_dir);
            cv::Mat undist;
            dataset_io_->undistortImage(img, undist);
            cv::imwrite(images_dir + std::to_string(k) + ".jpg", undist);
        }

        std::vector<float> zbuf(W * H, std::numeric_limits<float>::infinity());
        std::vector<pcl::PointXYZRGB> pixbuf(W * H);
        const float eps = 1e-6f;
        
        for (const auto& p : cloud_w_all_opt->points) {
            Eigen::Vector3d Xw(p.x, p.y, p.z);
        
            double u = 0, v = 0, zc = 0;
            if (!ProjectToImage(Rcw, tcw, Xw, &u, &v, &zc)) continue;
        
            int uu = static_cast<int>(std::round(u));
            int vv = static_cast<int>(std::round(v));
            if (uu < 0 || uu >= W || vv < 0 || vv >= H) continue;
        
            const int idx = vv * W + uu;
        
            // 只保留深度最近（zc越小越近）
            if (zc + eps < zbuf[idx]) {
                const cv::Vec3b bgr = img.at<cv::Vec3b>(vv, uu);
        
                pcl::PointXYZRGB cp;
                cp.x = p.x; cp.y = p.y; cp.z = p.z;
                cp.b = bgr[0]; cp.g = bgr[1]; cp.r = bgr[2];
        
                zbuf[idx]   = static_cast<float>(zc);
                pixbuf[idx] = cp;
            }
        }
        
        // 把每个像素最终留下的“最近点”收集到 colored 里
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored(new pcl::PointCloud<pcl::PointXYZRGB>());
        colored->reserve(W * H); // 粗略预留，可不写
        for (int i = 0; i < W * H; ++i) {
            if (std::isfinite(zbuf[i])) colored->push_back(pixbuf[i]);
        }

        *merged += *colored;

        std::vector<float> zbuf_b(W * H, std::numeric_limits<float>::infinity());
        std::vector<pcl::PointXYZRGB> pixbuf_b(W * H);
        const float eps_b = 1e-6f;
        for (const auto& p : cloud_w_all_orig->points) {
            Eigen::Vector3d Xw(p.x, p.y, p.z);

            double u = 0, v = 0, zc = 0;

            if (!ProjectToImage(Rcw_b, tcw_b, Xw, &u, &v, &zc)) continue;

            int uu = static_cast<int>(std::round(u));
            int vv = static_cast<int>(std::round(v));

            if (uu < 0 || uu >= W || vv < 0 || vv >= H) continue;

            const int idx = vv * W + uu;
        
            // 只保留深度最近（zc越小越近）
            if (zc + eps_b < zbuf_b[idx]) {
                const cv::Vec3b bgr = img.at<cv::Vec3b>(vv, uu);
        
                pcl::PointXYZRGB cp;
                cp.x = p.x; cp.y = p.y; cp.z = p.z;
                cp.b = bgr[0]; cp.g = bgr[1]; cp.r = bgr[2];
        
                zbuf_b[idx]   = static_cast<float>(zc);
                pixbuf_b[idx] = cp;
            }
        }
        // 把每个像素最终留下的“最近点”收集到 colored 里
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_b(new pcl::PointCloud<pcl::PointXYZRGB>());
        colored_b->reserve(W * H); // 粗略预留，可不写
        for (int i = 0; i < W * H; ++i) {
            if (std::isfinite(zbuf_b[i])) colored_b->push_back(pixbuf_b[i]);
        }
        *merged_b += *colored_b;
    }

    if(colmap_output_enable_)
    {
        std::cout << "[Colorize] Merged colored cloud size = " << merged->size() << "\n";
        down_sampling_voxel2(*merged, filter_size_points3D_);
        pcl::io::savePCDFileBinary(merged_pcd_path, *merged);
        std::cout << "[Colorize] Downsampled size = " << merged->size() << "\n";

        std::string sparse_dir = dataset_path_ + "Colmap/sparse/";
        if (!fs::exists(sparse_dir)) fs::create_directories(sparse_dir);
        fout_points_after.open(sparse_dir + "points3D.txt", std::ios::out);
        for (size_t i = 0; i < merged->size(); ++i) 
        {
            const auto& point = merged->points[i];
            fout_points_after << i << " "
                        << std::fixed << std::setprecision(6)
                        << point.x << " " << point.y << " " << point.z << " "
                        << static_cast<int>(point.r) << " "
                        << static_cast<int>(point.g) << " "
                        << static_cast<int>(point.b) << " "
                        << 0 << std::endl;
        }
    }

    pub_cloud_b_ = merged_b;
    pub_cloud_ = merged;

    std::vector<pcl::PointCloud<PointType>::Ptr>().swap(dataset_io_->pl_fulls_);
}

std::string LvbaSystem::getImagePath(double image_id) {
  return dataset_path_ + "all_image/" + std::to_string(image_id) + ".png";
}

std::string LvbaSystem::getPcdPath(double pcd_id) {
  return dataset_path_ + "all_pcd_body/" + std::to_string(pcd_id) + ".pcd";
}

void LvbaSystem::pubRGBCloud() {

    showTracksComparePCL();

    sensor_msgs::PointCloud2 output;
    down_sampling_voxel(*pub_cloud_, 0.01);
    pcl::toROSMsg(*pub_cloud_, output);
    output.header.frame_id = "map";
    output.header.stamp = ros::Time::now();

    cloud_pub_after_.publish(output);

    sensor_msgs::PointCloud2 output_b;
    down_sampling_voxel(*pub_cloud_b_, 0.01);
    pcl::toROSMsg(*pub_cloud_b_, output_b);
    output_b.header.frame_id = "map"; 
    output_b.header.stamp = ros::Time::now();

    cloud_pub_before_.publish(output_b);
}


}
