#ifndef POSE_VISUALIZER_H
#define POSE_VISUALIZER_H

// 3D Pose Visualizer - Dear ImGui + OpenGL + GLFW
// Visualizes camera pose optimization process with animation

#include <vector>
#include <string>
#include <memory>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

// Forward declarations to avoid including heavy headers
struct GLFWwindow;

namespace lvba {

// Simple Track structure for visualization (avoid dependency on full lvba_system)
struct TrackViz {
    Eigen::Vector3d Xw_before;  // 3D point before optimization
    Eigen::Vector3d Xw_after;   // 3D point after optimization
};

// Point correspondence for morph animation
struct PointCorrespondence {
    Eigen::Vector3f posBefore;  // Position before optimization
    Eigen::Vector3f posAfter;   // Position after optimization
    Eigen::Vector3f color;      // RGB color [0,1]
};

// Camera pose for visualization
struct CameraPose {
    Eigen::Matrix3d R;  // Rotation (world to camera)
    Eigen::Vector3d t;  // Translation
    
    // Get camera center in world coordinates
    Eigen::Vector3d getCenter() const {
        return -R.transpose() * t;
    }
};

// Point cloud display mode
enum class PointCloudMode {
    Before,       // Show cloud before optimization
    After,        // Show cloud after optimization
    Crossfade,    // Crossfade between before and after (alpha blending)
    PointMorph    // Each point moves from before position to after position
};

// Point cloud color mode
enum class ColorMode {
    RGB,         // Original RGB color
    IntensityJet // JET colormap based on intensity/height
};

// Rendering settings
struct RenderSettings {
    // Point cloud
    float pointSize = 2.0f;
    bool showPointCloud = true;
    float pointCloudAlpha = 0.8f;
    int pointCloudDownsample = 1;  // Show every N-th point (1 = no downsampling)
    PointCloudMode pointCloudMode = PointCloudMode::After;
    ColorMode colorMode = ColorMode::RGB;
    
    // Cameras
    float cameraSize = 0.3f;
    bool showCamerasBefore = true;
    bool showCamerasAfter = true;
    float cameraAlphaBefore = 0.4f;
    float cameraAlphaAfter = 0.9f;
    
    // Trajectory
    bool showTrajectory = true;
    float trajectoryWidth = 2.0f;
    
    // Grid
    bool showGrid = true;
    float gridSize = 50.0f;
    float gridSpacing = 1.0f;
    
    // Coordinate axes
    bool showAxes = true;
    float axesLength = 2.0f;
    
    // 3D Points (tracks)
    bool showTracks = false;
    float trackPointSize = 3.0f;
};

// Animation state
struct AnimationState {
    bool playing = false;
    float progress = 0.0f;  // 0.0 to 1.0
    float duration = 10.0f; // Total animation duration in seconds (3-30)
    bool loop = true;
};

// Orbit camera for 3D navigation
struct OrbitCamera {
    Eigen::Vector3d target = Eigen::Vector3d::Zero();
    float distance = 20.0f;
    float azimuth = 45.0f;   // Horizontal angle in degrees
    float elevation = 30.0f; // Vertical angle in degrees
    float fov = 45.0f;       // Field of view in degrees
    float nearPlane = 0.1f;
    float farPlane = 1000.0f;
    
    // Mouse interaction state
    bool rotating = false;
    bool panning = false;
    double lastMouseX = 0, lastMouseY = 0;
    
    // Smoothing
    float rotationSpeed = 0.3f;
    float panSpeed = 0.01f;
    float zoomSpeed = 1.1f;
    
    // Get view matrix
    Eigen::Matrix4f getViewMatrix() const;
    Eigen::Matrix4f getProjectionMatrix(float aspectRatio) const;
    Eigen::Vector3d getEyePosition() const;
};

class PoseVisualizer {
public:
    PoseVisualizer();
    ~PoseVisualizer();
    
    // Prevent copying
    PoseVisualizer(const PoseVisualizer&) = delete;
    PoseVisualizer& operator=(const PoseVisualizer&) = delete;
    
    // Initialize/shutdown
    bool init(int width = 1280, int height = 720, const char* title = "LVBA Pose Viewer");
    void shutdown();
    
    // Data setup
    void setPointCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud);
    void setPointCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloudBefore,
                       pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloudAfter);
    
    void setCameraPoses(const std::vector<CameraPose>& posesBefore,
                        const std::vector<CameraPose>& posesAfter);
    
    // Convenience: set poses from Eigen matrices directly
    void setCameraPoses(const std::vector<Eigen::Matrix3d>& RcwBefore,
                        const std::vector<Eigen::Vector3d>& tcwBefore,
                        const std::vector<Eigen::Matrix3d>& RcwAfter,
                        const std::vector<Eigen::Vector3d>& tcwAfter);
    
    void setTracks(const std::vector<TrackViz>& tracks);
    
    // Set point correspondences for morph animation
    void setPointCorrespondences(const std::vector<PointCorrespondence>& correspondences);
    
    // Set statistics for display
    void setStats(double convergenceRatio, double meanErrorBefore, double meanErrorAfter);
    
    // Main loop - blocks until window is closed
    void run();
    
    // Check if window should close
    bool shouldClose() const;
    
    // Single frame update (for non-blocking usage)
    void pollEvents();
    void renderFrame();
    
private:
    // OpenGL initialization
    bool initOpenGL();
    void createShaders();
    void createBuffers();
    
    // Rendering
    void renderScene();
    void renderPointCloud();
    void renderCameraFrustum(const CameraPose& pose, float r, float g, float b, float alpha);
    void renderCameraFrustums(float t);
    void renderTrajectory(float t);
    void renderGrid();
    void renderAxes();
    void renderTracks(float t);
    
    // ImGui
    void renderImGui();
    void renderControlPanel();
    void renderStatsPanel();
    
    // Input handling
    static void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
    static void cursorPosCallback(GLFWwindow* window, double xpos, double ypos);
    static void scrollCallback(GLFWwindow* window, double xoffset, double yoffset);
    static void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
    static void framebufferSizeCallback(GLFWwindow* window, int width, int height);
    
    // Interpolation helpers
    CameraPose interpolatePose(const CameraPose& a, const CameraPose& b, float t) const;
    Eigen::Vector3d interpolatePoint(const Eigen::Vector3d& a, const Eigen::Vector3d& b, float t) const;
    
    // Auto-fit camera to scene
    void fitCameraToScene();
    
private:
    GLFWwindow* window_ = nullptr;
    int windowWidth_ = 1280;
    int windowHeight_ = 720;
    bool initialized_ = false;
    
    // Data
    std::vector<float> pointCloudVertices_;  // x,y,z,r,g,b interleaved (RGB mode, before)
    std::vector<float> pointCloudVerticesAfter_;  // RGB mode, after
    std::vector<float> pointCloudVerticesJet_;    // JET colormap mode, before
    std::vector<float> pointCloudVerticesJetAfter_;  // JET colormap mode, after
    size_t numPoints_ = 0;
    size_t numPointsAfter_ = 0;
    
    // Helper function for JET colormap
    static void jetColormap(float value, float& r, float& g, float& b);
    
    std::vector<CameraPose> posesBefore_;
    std::vector<CameraPose> posesAfter_;
    
    std::vector<TrackViz> tracks_;
    
    // Point correspondences for morph animation
    std::vector<PointCorrespondence> pointCorrespondences_;
    std::vector<float> morphVertices_;  // Interpolated vertices for current frame
    unsigned int morphVAO_ = 0, morphVBO_ = 0;
    
    // OpenGL objects
    unsigned int pointCloudVAO_ = 0, pointCloudVBO_ = 0;
    unsigned int pointCloudAfterVAO_ = 0, pointCloudAfterVBO_ = 0;
    unsigned int lineVAO_ = 0, lineVBO_ = 0;
    unsigned int shaderProgram_ = 0;
    
    // State
    RenderSettings settings_;
    AnimationState animation_;
    OrbitCamera camera_;
    
    // Statistics
    double convergenceRatio_ = 0.0;
    double meanErrorBefore_ = 0.0;
    double meanErrorAfter_ = 0.0;
    size_t numCameras_ = 0;
    size_t numTracks_ = 0;
    
    // Timing
    double lastFrameTime_ = 0.0;
    float deltaTime_ = 0.0f;
};

} // namespace lvba

#endif // POSE_VISUALIZER_H
