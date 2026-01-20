#include "dataset_io.h"

namespace lvba {

DatasetIO::DatasetIO(ros::NodeHandle& nh) 
{    
    readParameters(nh);

    if (loadDataset()) {
        std::cout << "[DatasetIO] Data loaded successfully" << std::endl;
    } else {
        std::cerr << "[DatasetIO] Failed to load dataset" << std::endl;
    }

    undist_map1_ = cv::Mat(height_, width_, CV_16SC2);
    undist_map2_ = cv::Mat(height_, width_, CV_16UC1);
    cvK_ = (cv::Mat_<float>(3, 3) << fx_, 0.0, cx_, 0.0, fy_, cy_, 0.0, 0.0, 1.0);
    cvD_ = (cv::Mat_<float>(1, 5) << k1_, k2_, p1_, p2_, 0);
    cv::initUndistortRectifyMap(cvK_, cvD_, cv::Mat_<double>::eye(3,3), cvK_,
                              cv::Size(width_, height_), CV_16SC2, undist_map1_, undist_map2_);
}

void DatasetIO::undistortImage(const cv::Mat& raw, cv::Mat& rectified)
{
    cv::remap(raw, rectified, undist_map1_, undist_map2_, cv::INTER_LINEAR);
}

void DatasetIO::readParameters(ros::NodeHandle &nh)
{
    nh.param<std::string>("data_config/data_path", dataset_path_, "dataset/cbd_new/");
    nh.param<std::string>("data_config/colmap_db_path", colmap_db_path_, "");
    nh.param<int>("data_config/image_sample_step", image_stride_, 10);
    nh.param<int>("cam_model/cam_width", width_, 1280);
    nh.param<int>("cam_model/cam_height", height_, 1024);
    nh.param<double>("cam_model/scale", resize_scale_, 0.5);
    nh.param<double>("cam_model/cam_fx", fx_, 1293.56944);
    nh.param<double>("cam_model/cam_fy", fy_, 1293.3155);
    nh.param<double>("cam_model/cam_cx", cx_, 626.91359);
    nh.param<double>("cam_model/cam_cy", cy_, 522.799224);
    nh.param<double>("cam_model/cam_d0", k1_, -0.076160);
    nh.param<double>("cam_model/cam_d1", k2_, 0.123001);
    nh.param<double>("cam_model/cam_d2", p1_, -0.00113);
    nh.param<double>("cam_model/cam_d3", p2_, 0.000251);
    nh.param<std::vector<double>>("extrin_calib/extrinsic_T", extrinT_, std::vector<double>());
    nh.param<std::vector<double>>("extrin_calib/extrinsic_R", extrinR_, std::vector<double>());
    nh.param<std::vector<double>>("extrin_calib/Pcl", cameraextrinT_, std::vector<double>());
    nh.param<std::vector<double>>("extrin_calib/Rcl", cameraextrinR_, std::vector<double>());

    nh.param<bool>("window_ba/enable", window_ba_enable_, true);
    nh.param<int>("window_ba/size", window_ba_size_, 10);
    nh.param<double>("window_ba/anchor_leaf_size", anchor_leaf_size_, 0.1);
    nh.param<bool>("window_ba/use_window_ba_rel", use_window_ba_rel_, false);
    nh.param<double>("BALM_stage1/root_voxel_size", stage1_root_voxel_size_, 0.5);
    nh.param<bool>("BALM_stage1/enable", stage1_enable_, true);
    nh.param<double>("BALM_stage2/root_voxel_size", stage2_root_voxel_size_, stage1_root_voxel_size_);
    nh.param<std::vector<float>>("BALM_stage1/eigen_ratio_array", stage1_eigen_ratio_array_, stage1_eigen_ratio_array_);
    nh.param<std::vector<float>>("BALM_stage2/eigen_ratio_array", stage2_eigen_ratio_array_, stage2_eigen_ratio_array_);

    width_  = static_cast<int>(std::lround(width_  * resize_scale_));
    height_ = static_cast<int>(std::lround(height_ * resize_scale_));
    fx_ *= resize_scale_; fy_ *= resize_scale_;
    cx_ *= resize_scale_; cy_ *= resize_scale_;

    // 如果dataset_path_不是绝对路径，则添加ROOT_DIR前缀
    if (dataset_path_.empty() || dataset_path_[0] != '/') {
        dataset_path_ = std::string(ROOT_DIR) + dataset_path_;
    }
    colmap_db_path_ = dataset_path_ + colmap_db_path_;
}

bool DatasetIO::loadDataset()
{
  if (!handleImages()) return false;
  if (!handleCamPoses()) return false;
  if (!handleLidarPoses()) return false; 
  if (!handleBodyPoints()) return false;
  return true;
}

bool DatasetIO::handleImages() {
  namespace fs = std::filesystem;
  const std::string images_path = dataset_path_ + "all_image";

  if (!fs::exists(images_path)) {
    std::cerr << "[handleImages] Path does not exist: " << images_path << "\n";
    return false;
  }
  if (!fs::is_directory(images_path)) {
    std::cerr << "[handleImages] Not a directory: " << images_path << "\n";
    return false;
  }
  if (image_stride_ == 0) {
    std::cerr << "[handleImages] image_stride_ cannot be 0\n";
    return false;
  }

  // match int/float in filename
  std::regex re(R"(([0-9]+(?:\.[0-9]+)?))");

  images_ids_.clear();
  for (const auto& entry : fs::directory_iterator(images_path)) {
    if (!entry.is_regular_file()) continue;

    const auto path = entry.path();
    const std::string ext = path.extension().string();
    if (ext != ".png" && ext != ".jpg" && ext != ".jpeg" && ext != ".bmp") continue;

    const std::string filename = path.filename().string();
    std::smatch match;
    if (!std::regex_search(filename, match, re)) {
      std::cerr << "[handleImages] Skip file without numeric id: " << filename << "\n";
      continue;
    }
    images_ids_.push_back(std::stod(match[1].str()));
  }

  if (images_ids_.empty()) {
    std::cerr << "[handleImages] No valid image files found in " << images_path << "\n";
    return false;
  }

  std::sort(images_ids_.begin(), images_ids_.end());

  // downsample
  std::vector<double> down;
  down.reserve((images_ids_.size() + image_stride_ - 1) / image_stride_);
  for (size_t i = 0; i < images_ids_.size(); i += image_stride_) down.push_back(images_ids_[i]);
  images_ids_.swap(down);

  std::cout << "[handleImages] Loaded " << images_ids_.size()
            << " images (take 1 every " << image_stride_ << ") from " << images_path
            << "\n";
  return true;
}

bool DatasetIO::loadPosesTUM(const std::string& file,
                            size_t STRIDE,
                            std::vector<Sophus::SE3>& poses_out) {
  std::ifstream fin(file);
  if (!fin.is_open()) {
    std::cerr << "[loadPosesTUM] Failed to open: " << file << "\n";
    return false;
  }
  if (STRIDE == 0) {
    std::cerr << "[loadPosesTUM] Invalid stride=0 in " << file << "\n";
    return false;
  }

  poses_out.clear();

  std::string line;
  size_t line_no = 0;
  size_t valid_idx = 0;
  size_t selected = 0;

  while (std::getline(fin, line)) {
    ++line_no;
    if (line.empty() || line[0] == '#') continue;

    std::istringstream iss(line);
    double timestamp, tx, ty, tz, qx, qy, qz, qw;
    if (!(iss >> timestamp >> tx >> ty >> tz >> qx >> qy >> qz >> qw)) {
      std::cerr << "[loadPosesTUM] Parse fail at line " << line_no << " in " << file << "\n";
      continue;
    }

    // offset fixed to 0
    if ((valid_idx % STRIDE) == 0) {
      Eigen::Quaterniond q(qw, qx, qy, qz);
      q.normalize();
      Eigen::Vector3d t(tx, ty, tz);
      poses_out.emplace_back(q, Eigen::Vector3d(tx, ty, tz));
      ++selected;
    }
    ++valid_idx;
  }

  if (poses_out.empty()) {
    std::cerr << "[loadPosesTUM] No poses loaded from: " << file << "\n";
    return false;
  }

  std::cout << "[loadPosesTUM] Loaded poses=" << poses_out.size() << " (selected=" << selected
            << ", parsed=" << valid_idx << ") from " << file << " (stride=" << STRIDE
            << ")\n";
  return true;
}

bool DatasetIO::handleLidarPoses() 
{
  const std::string file = dataset_path_ + "all_pcd_body/lidar_poses.txt";

  if (!loadPosesTUM(file, 1, lidar_poses_)) {
    std::cerr << "[handleLidarPoses] handleLidarPoses failed: " << file << "\n";
    return false;
  }
  return true;
}

bool DatasetIO::handleCamPoses() 
{
  const std::string file = dataset_path_ + "all_image/image_poses.txt";
  if (!loadPosesTUM(file, image_stride_, image_poses_)) {
    std::cerr << "[handleLidarPoses] handleCamPoses failed: " << file << "\n";
    return false;
  }
  if (image_poses_.size() != images_ids_.size()) {
    std::cerr << "[handleLidarPoses] cam pose count != image count: cam_poses=" << image_poses_.size()
              << " images=" << images_ids_.size() << "\n";
    return false;
  }
  return true;
}

bool DatasetIO::handleBodyPoints() {
  namespace fs = std::filesystem;

  const std::string pcd_dir = dataset_path_ + "all_pcd_body";

  if (!fs::exists(pcd_dir) || !fs::is_directory(pcd_dir)) {
    std::cerr << "[handleBodyPoints] pcd dir missing: " << pcd_dir << "\n";
    return false;
  }

  std::vector<std::pair<double, std::string>> pcds;
  pcds.reserve(4096);

  for (const auto& e : fs::directory_iterator(pcd_dir)) {
    if (!e.is_regular_file() || e.path().extension() != ".pcd") continue;

    double ts = 0.0;
    if (!parseTimestampFromName(e.path().filename().string(), ts)) {
      std::cerr << "[handleBodyPoints] bad pcd name: " << e.path() << "\n";
      continue;
    }
    pcds.emplace_back(ts, e.path().string());
  }

  if (pcds.empty()) {
    std::cerr << "[handleBodyPoints] no pcd files in: " << pcd_dir << "\n";
    return false;
  }

  std::sort(pcds.begin(), pcds.end(), [](const auto& a, const auto& b) { return a.first < b.first; });

  std::vector<double> pcd_ts;
  pcd_ts.reserve(pcds.size());
  for (const auto& kv : pcds) pcd_ts.push_back(kv.first);

  // if (pcds.size() != lidar_poses_.size()) {
  //   std::cerr << "[Points+VoxelId] size mismatch: pcds=" << pcds.size()
  //             << " lidar_poses=" << lidar_poses_.size() << "\n";
  //   return false;
  // }

  for (size_t m = 0; m < lidar_poses_.size(); ++m) {
    IMUST curr;
    curr.R = lidar_poses_[m].rotation_matrix(); curr.p = lidar_poses_[m].translation(); curr.t = pcd_ts[m];
    x_buf_.push_back(curr);
    // std::cout << "Lidar Pose " << i << ": " << lidar_poses_[i].log().transpose() << "\n";
  }
  x_buf_before_ = x_buf_;  // 保留原始未修改的轨迹

  // PointCloudXYZI::Ptr pl_world(new PointCloudXYZI); pl_world->reserve(2000000);

  for (size_t i = 0; i < pcds.size(); ++i) 
  {
      const std::string& pcd_path = pcds[i].second;

      pcl::PointCloud<PointType>::Ptr pl_ptr_body(new pcl::PointCloud<PointType>());
      pcl::PointCloud<pcl::PointXYZI> pl_tem;

      if (pcl::io::loadPCDFile(pcd_path, pl_tem) != 0) {
        std::cerr << "[handleBodyPoints] failed to load pcd: " << pcd_path << "\n";
        continue;
      }

      for(pcl::PointXYZI &pp: pl_tem.points)
      {
        PointType ap;
        ap.x = pp.x; ap.y = pp.y; ap.z = pp.z;
        ap.intensity = pp.intensity;
        pl_ptr_body->push_back(ap);
      }

      pl_fulls_.push_back(pl_ptr_body);

      // pcl::PointCloud<PointType>::Ptr pl_ptr_world(new pcl::PointCloud<PointType>(*pl_ptr_body));
      // transformPointBodyToWorld(pl_ptr_world, lidar_poses_[i]);
      // *pl_world += *pl_ptr_world;
  }
  // cloud_ = pl_world;
  // cout << "[Points+VoxelId] Merged global cloud points=" << cloud_->size() << "\n";
  
  // save merged cloud for visualization
  // const std::string merged_path = dataset_path_ + "merged_global_cloud.pcd";
  // if (pcl::io::savePCDFileBinaryCompressed(merged_path, *cloud_) != 0) {
  //     std::cerr << "[Points+VoxelId] failed to save merged cloud to: " << merged_path << "\n";
  // } else {
  //     std::cout << "[Points+VoxelId] saved merged global cloud to: " << merged_path << "\n";
  // }
  return true;
}

}
