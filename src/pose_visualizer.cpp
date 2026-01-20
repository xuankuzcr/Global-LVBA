// 3D Pose Visualizer Implementation
// Dear ImGui + OpenGL + GLFW

#include "pose_visualizer.h"

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <iostream>
#include <cmath>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace lvba {

// ============================================================================
// JET Colormap
// ============================================================================

void PoseVisualizer::jetColormap(float value, float& r, float& g, float& b) {
    // Clamp value to [0, 1]
    value = std::max(0.0f, std::min(1.0f, value));
    
    // JET colormap: blue -> cyan -> green -> yellow -> red
    if (value < 0.25f) {
        r = 0.0f;
        g = 4.0f * value;
        b = 1.0f;
    } else if (value < 0.5f) {
        r = 0.0f;
        g = 1.0f;
        b = 1.0f - 4.0f * (value - 0.25f);
    } else if (value < 0.75f) {
        r = 4.0f * (value - 0.5f);
        g = 1.0f;
        b = 0.0f;
    } else {
        r = 1.0f;
        g = 1.0f - 4.0f * (value - 0.75f);
        b = 0.0f;
    }
}

// ============================================================================
// Shader sources
// ============================================================================

static const char* vertexShaderSource = R"(
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aColor;

uniform mat4 uMVP;
uniform float uPointSize;

out vec3 vColor;

void main() {
    gl_Position = uMVP * vec4(aPos, 1.0);
    gl_PointSize = uPointSize;
    vColor = aColor;
}
)";

static const char* fragmentShaderSource = R"(
#version 330 core
in vec3 vColor;
uniform float uAlpha;
out vec4 FragColor;

void main() {
    FragColor = vec4(vColor, uAlpha);
}
)";

// ============================================================================
// OrbitCamera implementation
// ============================================================================

Eigen::Matrix4f OrbitCamera::getViewMatrix() const {
    Eigen::Vector3d eye = getEyePosition();
    Eigen::Vector3d forward = (target - eye).normalized();
    Eigen::Vector3d right = forward.cross(Eigen::Vector3d::UnitZ()).normalized();
    
    // Handle case when looking straight up/down
    if (right.norm() < 0.001) {
        right = Eigen::Vector3d::UnitX();
    }
    Eigen::Vector3d up = right.cross(forward).normalized();
    
    Eigen::Matrix4f view = Eigen::Matrix4f::Identity();
    view(0, 0) = (float)right.x();    view(0, 1) = (float)right.y();    view(0, 2) = (float)right.z();
    view(1, 0) = (float)up.x();       view(1, 1) = (float)up.y();       view(1, 2) = (float)up.z();
    view(2, 0) = (float)-forward.x(); view(2, 1) = (float)-forward.y(); view(2, 2) = (float)-forward.z();
    view(0, 3) = (float)-right.dot(eye);
    view(1, 3) = (float)-up.dot(eye);
    view(2, 3) = (float)forward.dot(eye);
    
    return view;
}

Eigen::Matrix4f OrbitCamera::getProjectionMatrix(float aspectRatio) const {
    float fovRad = fov * (float)M_PI / 180.0f;
    float tanHalfFov = std::tan(fovRad / 2.0f);
    
    Eigen::Matrix4f proj = Eigen::Matrix4f::Zero();
    proj(0, 0) = 1.0f / (aspectRatio * tanHalfFov);
    proj(1, 1) = 1.0f / tanHalfFov;
    proj(2, 2) = -(farPlane + nearPlane) / (farPlane - nearPlane);
    proj(2, 3) = -2.0f * farPlane * nearPlane / (farPlane - nearPlane);
    proj(3, 2) = -1.0f;
    
    return proj;
}

Eigen::Vector3d OrbitCamera::getEyePosition() const {
    float azimuthRad = azimuth * (float)M_PI / 180.0f;
    float elevationRad = elevation * (float)M_PI / 180.0f;
    
    double x = distance * std::cos(elevationRad) * std::cos(azimuthRad);
    double y = distance * std::cos(elevationRad) * std::sin(azimuthRad);
    double z = distance * std::sin(elevationRad);
    
    return target + Eigen::Vector3d(x, y, z);
}

// ============================================================================
// PoseVisualizer implementation
// ============================================================================

PoseVisualizer::PoseVisualizer() = default;

PoseVisualizer::~PoseVisualizer() {
    shutdown();
}

bool PoseVisualizer::init(int width, int height, const char* title) {
    windowWidth_ = width;
    windowHeight_ = height;
    
    // Initialize GLFW
    if (!glfwInit()) {
        std::cerr << "[PoseVisualizer] Failed to initialize GLFW" << std::endl;
        return false;
    }
    
    // OpenGL 3.3 Core Profile
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_SAMPLES, 4);  // MSAA
    
    // Create window
    window_ = glfwCreateWindow(width, height, title, nullptr, nullptr);
    if (!window_) {
        std::cerr << "[PoseVisualizer] Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return false;
    }
    
    glfwMakeContextCurrent(window_);
    glfwSwapInterval(1);  // VSync
    
    // Store this pointer for callbacks
    glfwSetWindowUserPointer(window_, this);
    
    // Set callbacks
    glfwSetMouseButtonCallback(window_, mouseButtonCallback);
    glfwSetCursorPosCallback(window_, cursorPosCallback);
    glfwSetScrollCallback(window_, scrollCallback);
    glfwSetKeyCallback(window_, keyCallback);
    glfwSetFramebufferSizeCallback(window_, framebufferSizeCallback);
    
    // Initialize GLAD
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "[PoseVisualizer] Failed to initialize GLAD" << std::endl;
        glfwDestroyWindow(window_);
        glfwTerminate();
        return false;
    }
    
    // Initialize OpenGL settings
    if (!initOpenGL()) {
        return false;
    }
    
    // Initialize ImGui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    
    // Style
    ImGui::StyleColorsDark();
    ImGuiStyle& style = ImGui::GetStyle();
    style.WindowRounding = 8.0f;
    style.FrameRounding = 4.0f;
    style.GrabRounding = 4.0f;
    style.Colors[ImGuiCol_WindowBg] = ImVec4(0.1f, 0.1f, 0.12f, 0.94f);
    style.Colors[ImGuiCol_Header] = ImVec4(0.2f, 0.4f, 0.6f, 0.8f);
    style.Colors[ImGuiCol_HeaderHovered] = ImVec4(0.3f, 0.5f, 0.7f, 0.8f);
    style.Colors[ImGuiCol_Button] = ImVec4(0.2f, 0.4f, 0.6f, 0.8f);
    style.Colors[ImGuiCol_ButtonHovered] = ImVec4(0.3f, 0.5f, 0.7f, 1.0f);
    style.Colors[ImGuiCol_SliderGrab] = ImVec4(0.3f, 0.6f, 0.9f, 1.0f);
    
    // Platform/Renderer backends
    ImGui_ImplGlfw_InitForOpenGL(window_, true);
    ImGui_ImplOpenGL3_Init("#version 330");
    
    lastFrameTime_ = glfwGetTime();
    initialized_ = true;
    
    std::cout << "[PoseVisualizer] Initialized successfully (" << width << "x" << height << ")" << std::endl;
    return true;
}

void PoseVisualizer::shutdown() {
    if (!initialized_) return;
    
    // Cleanup OpenGL
    if (pointCloudVAO_) glDeleteVertexArrays(1, &pointCloudVAO_);
    if (pointCloudVBO_) glDeleteBuffers(1, &pointCloudVBO_);
    if (pointCloudAfterVAO_) glDeleteVertexArrays(1, &pointCloudAfterVAO_);
    if (pointCloudAfterVBO_) glDeleteBuffers(1, &pointCloudAfterVBO_);
    if (lineVAO_) glDeleteVertexArrays(1, &lineVAO_);
    if (lineVBO_) glDeleteBuffers(1, &lineVBO_);
    if (morphVAO_) glDeleteVertexArrays(1, &morphVAO_);
    if (morphVBO_) glDeleteBuffers(1, &morphVBO_);
    if (shaderProgram_) glDeleteProgram(shaderProgram_);
    
    // Cleanup ImGui
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    
    // Cleanup GLFW
    if (window_) {
        glfwDestroyWindow(window_);
        window_ = nullptr;
    }
    glfwTerminate();
    
    initialized_ = false;
    std::cout << "[PoseVisualizer] Shutdown complete" << std::endl;
}

bool PoseVisualizer::initOpenGL() {
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_PROGRAM_POINT_SIZE);
    glEnable(GL_MULTISAMPLE);
    glEnable(GL_LINE_SMOOTH);
    
    createShaders();
    createBuffers();
    
    return true;
}

void PoseVisualizer::createShaders() {
    // Compile vertex shader
    unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, nullptr);
    glCompileShader(vertexShader);
    
    int success;
    char infoLog[512];
    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(vertexShader, 512, nullptr, infoLog);
        std::cerr << "[PoseVisualizer] Vertex shader compilation failed: " << infoLog << std::endl;
    }
    
    // Compile fragment shader
    unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, nullptr);
    glCompileShader(fragmentShader);
    
    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(fragmentShader, 512, nullptr, infoLog);
        std::cerr << "[PoseVisualizer] Fragment shader compilation failed: " << infoLog << std::endl;
    }
    
    // Link program
    shaderProgram_ = glCreateProgram();
    glAttachShader(shaderProgram_, vertexShader);
    glAttachShader(shaderProgram_, fragmentShader);
    glLinkProgram(shaderProgram_);
    
    glGetProgramiv(shaderProgram_, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(shaderProgram_, 512, nullptr, infoLog);
        std::cerr << "[PoseVisualizer] Shader program linking failed: " << infoLog << std::endl;
    }
    
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
}

void PoseVisualizer::createBuffers() {
    // Point cloud VAO/VBO
    glGenVertexArrays(1, &pointCloudVAO_);
    glGenBuffers(1, &pointCloudVBO_);
    
    glGenVertexArrays(1, &pointCloudAfterVAO_);
    glGenBuffers(1, &pointCloudAfterVBO_);
    
    // Morph animation VAO/VBO
    glGenVertexArrays(1, &morphVAO_);
    glGenBuffers(1, &morphVBO_);
    
    // Line VAO/VBO (for dynamic lines)
    glGenVertexArrays(1, &lineVAO_);
    glGenBuffers(1, &lineVBO_);
}

void PoseVisualizer::setPointCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud) {
    setPointCloud(cloud, nullptr);
}

void PoseVisualizer::setPointCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloudBefore,
                                    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloudAfter) {
    pointCloudVertices_.clear();
    pointCloudVerticesAfter_.clear();
    pointCloudVerticesJet_.clear();
    pointCloudVerticesJetAfter_.clear();
    numPoints_ = 0;
    numPointsAfter_ = 0;
    
    // Helper to compute Z range for JET colormap
    auto computeZRange = [](pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, float& zMin, float& zMax) {
        zMin = std::numeric_limits<float>::max();
        zMax = std::numeric_limits<float>::lowest();
        for (const auto& pt : *cloud) {
            if (!std::isfinite(pt.z)) continue;
            zMin = std::min(zMin, pt.z);
            zMax = std::max(zMax, pt.z);
        }
        if (zMax - zMin < 0.001f) {
            zMin -= 0.5f;
            zMax += 0.5f;
        }
    };
    
    // Process "before" cloud
    if (cloudBefore && !cloudBefore->empty()) {
        float zMin, zMax;
        computeZRange(cloudBefore, zMin, zMax);
        float zRange = zMax - zMin;
        
        pointCloudVertices_.reserve(cloudBefore->size() * 6);
        pointCloudVerticesJet_.reserve(cloudBefore->size() * 6);
        
        for (const auto& pt : *cloudBefore) {
            if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z)) continue;
            
            // RGB colors
            pointCloudVertices_.push_back(pt.x);
            pointCloudVertices_.push_back(pt.y);
            pointCloudVertices_.push_back(pt.z);
            pointCloudVertices_.push_back(pt.r / 255.0f);
            pointCloudVertices_.push_back(pt.g / 255.0f);
            pointCloudVertices_.push_back(pt.b / 255.0f);
            
            // JET colors based on height
            float normalizedZ = (pt.z - zMin) / zRange;
            float jr, jg, jb;
            jetColormap(normalizedZ, jr, jg, jb);
            
            pointCloudVerticesJet_.push_back(pt.x);
            pointCloudVerticesJet_.push_back(pt.y);
            pointCloudVerticesJet_.push_back(pt.z);
            pointCloudVerticesJet_.push_back(jr);
            pointCloudVerticesJet_.push_back(jg);
            pointCloudVerticesJet_.push_back(jb);
        }
        numPoints_ = pointCloudVertices_.size() / 6;
        
        // Upload RGB to GPU
        glBindVertexArray(pointCloudVAO_);
        glBindBuffer(GL_ARRAY_BUFFER, pointCloudVBO_);
        glBufferData(GL_ARRAY_BUFFER, pointCloudVertices_.size() * sizeof(float),
                     pointCloudVertices_.data(), GL_STATIC_DRAW);
        
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
        glEnableVertexAttribArray(1);
        
        glBindVertexArray(0);
        
        std::cout << "[PoseVisualizer] Point cloud (before) loaded: " << numPoints_ << " points" << std::endl;
        std::cout << "[PoseVisualizer] Z range: " << zMin << " to " << zMax << std::endl;
    }
    
    // Process "after" cloud (if different)
    if (cloudAfter && !cloudAfter->empty()) {
        float zMin, zMax;
        computeZRange(cloudAfter, zMin, zMax);
        float zRange = zMax - zMin;
        
        pointCloudVerticesAfter_.reserve(cloudAfter->size() * 6);
        pointCloudVerticesJetAfter_.reserve(cloudAfter->size() * 6);
        
        for (const auto& pt : *cloudAfter) {
            if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z)) continue;
            
            // RGB colors
            pointCloudVerticesAfter_.push_back(pt.x);
            pointCloudVerticesAfter_.push_back(pt.y);
            pointCloudVerticesAfter_.push_back(pt.z);
            pointCloudVerticesAfter_.push_back(pt.r / 255.0f);
            pointCloudVerticesAfter_.push_back(pt.g / 255.0f);
            pointCloudVerticesAfter_.push_back(pt.b / 255.0f);
            
            // JET colors based on height
            float normalizedZ = (pt.z - zMin) / zRange;
            float jr, jg, jb;
            jetColormap(normalizedZ, jr, jg, jb);
            
            pointCloudVerticesJetAfter_.push_back(pt.x);
            pointCloudVerticesJetAfter_.push_back(pt.y);
            pointCloudVerticesJetAfter_.push_back(pt.z);
            pointCloudVerticesJetAfter_.push_back(jr);
            pointCloudVerticesJetAfter_.push_back(jg);
            pointCloudVerticesJetAfter_.push_back(jb);
        }
        numPointsAfter_ = pointCloudVerticesAfter_.size() / 6;
        
        glBindVertexArray(pointCloudAfterVAO_);
        glBindBuffer(GL_ARRAY_BUFFER, pointCloudAfterVBO_);
        glBufferData(GL_ARRAY_BUFFER, pointCloudVerticesAfter_.size() * sizeof(float),
                     pointCloudVerticesAfter_.data(), GL_STATIC_DRAW);
        
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
        glEnableVertexAttribArray(1);
        
        glBindVertexArray(0);
        
        std::cout << "[PoseVisualizer] Point cloud (after) loaded: " << numPointsAfter_ << " points" << std::endl;
    }
    
    fitCameraToScene();
}

void PoseVisualizer::setCameraPoses(const std::vector<CameraPose>& posesBefore,
                                     const std::vector<CameraPose>& posesAfter) {
    posesBefore_ = posesBefore;
    posesAfter_ = posesAfter;
    numCameras_ = posesBefore.size();
    
    std::cout << "[PoseVisualizer] Camera poses loaded: " << numCameras_ << " cameras" << std::endl;
    
    if (numPoints_ == 0) {
        fitCameraToScene();
    }
}

void PoseVisualizer::setCameraPoses(const std::vector<Eigen::Matrix3d>& RcwBefore,
                                     const std::vector<Eigen::Vector3d>& tcwBefore,
                                     const std::vector<Eigen::Matrix3d>& RcwAfter,
                                     const std::vector<Eigen::Vector3d>& tcwAfter) {
    size_t n = std::min({RcwBefore.size(), tcwBefore.size(), RcwAfter.size(), tcwAfter.size()});
    
    posesBefore_.resize(n);
    posesAfter_.resize(n);
    
    for (size_t i = 0; i < n; ++i) {
        posesBefore_[i].R = RcwBefore[i];
        posesBefore_[i].t = tcwBefore[i];
        posesAfter_[i].R = RcwAfter[i];
        posesAfter_[i].t = tcwAfter[i];
    }
    
    numCameras_ = n;
    std::cout << "[PoseVisualizer] Camera poses loaded: " << numCameras_ << " cameras" << std::endl;
    
    if (numPoints_ == 0) {
        fitCameraToScene();
    }
}

void PoseVisualizer::setTracks(const std::vector<TrackViz>& tracks) {
    tracks_ = tracks;
    numTracks_ = tracks.size();
    std::cout << "[PoseVisualizer] Tracks loaded: " << numTracks_ << " tracks" << std::endl;
}

void PoseVisualizer::setPointCorrespondences(const std::vector<PointCorrespondence>& correspondences) {
    pointCorrespondences_ = correspondences;
    
    // Pre-allocate morph vertices buffer (x,y,z,r,g,b per point)
    morphVertices_.resize(correspondences.size() * 6);
    
    std::cout << "[PoseVisualizer] Point correspondences loaded: " << correspondences.size() << " points for morph animation" << std::endl;
}

void PoseVisualizer::setStats(double convergenceRatio, double meanErrorBefore, double meanErrorAfter) {
    convergenceRatio_ = convergenceRatio;
    meanErrorBefore_ = meanErrorBefore;
    meanErrorAfter_ = meanErrorAfter;
}

void PoseVisualizer::fitCameraToScene() {
    if (posesBefore_.empty() && numPoints_ == 0) return;
    
    Eigen::Vector3d minPt(1e10, 1e10, 1e10);
    Eigen::Vector3d maxPt(-1e10, -1e10, -1e10);
    
    // Include camera positions
    for (const auto& pose : posesBefore_) {
        Eigen::Vector3d c = pose.getCenter();
        minPt = minPt.cwiseMin(c);
        maxPt = maxPt.cwiseMax(c);
    }
    for (const auto& pose : posesAfter_) {
        Eigen::Vector3d c = pose.getCenter();
        minPt = minPt.cwiseMin(c);
        maxPt = maxPt.cwiseMax(c);
    }
    
    // Include point cloud (sample)
    for (size_t i = 0; i < pointCloudVertices_.size(); i += 6 * 100) {
        Eigen::Vector3d pt(pointCloudVertices_[i], pointCloudVertices_[i+1], pointCloudVertices_[i+2]);
        minPt = minPt.cwiseMin(pt);
        maxPt = maxPt.cwiseMax(pt);
    }
    
    camera_.target = (minPt + maxPt) * 0.5;
    camera_.distance = (float)((maxPt - minPt).norm() * 1.5);
    if (camera_.distance < 1.0f) camera_.distance = 20.0f;
    
    std::cout << "[PoseVisualizer] Camera fitted to scene, distance=" << camera_.distance << std::endl;
}

bool PoseVisualizer::shouldClose() const {
    return window_ && glfwWindowShouldClose(window_);
}

void PoseVisualizer::pollEvents() {
    glfwPollEvents();
}

void PoseVisualizer::run() {
    if (!initialized_) {
        std::cerr << "[PoseVisualizer] Not initialized!" << std::endl;
        return;
    }
    
    std::cout << "[PoseVisualizer] Starting render loop..." << std::endl;
    std::cout << "  Mouse controls:" << std::endl;
    std::cout << "    - Left button: rotate" << std::endl;
    std::cout << "    - Right button: pan" << std::endl;
    std::cout << "    - Scroll: zoom" << std::endl;
    std::cout << "  Keyboard shortcuts:" << std::endl;
    std::cout << "    - Space: play/pause animation" << std::endl;
    std::cout << "    - 1/2/3/4: switch cloud (Before/After/Crossfade/Morph)" << std::endl;
    std::cout << "    - C: toggle color mode (RGB/JET)" << std::endl;
    std::cout << "    - P: toggle point cloud" << std::endl;
    std::cout << "    - T: toggle trajectory" << std::endl;
    std::cout << "    - G: toggle grid" << std::endl;
    std::cout << "    - R: reset view" << std::endl;
    std::cout << "    - ESC: close" << std::endl;
    std::cout << "  Animation modes:" << std::endl;
    std::cout << "    - Crossfade (3): alpha blend between clouds" << std::endl;
    std::cout << "    - Morph (4): each point moves from before to after" << std::endl;
    
    while (!glfwWindowShouldClose(window_)) {
        pollEvents();
        renderFrame();
    }
}

void PoseVisualizer::renderFrame() {
    // Update timing
    double currentTime = glfwGetTime();
    deltaTime_ = (float)(currentTime - lastFrameTime_);
    lastFrameTime_ = currentTime;
    
    // Update animation - use duration to calculate speed
    if (animation_.playing) {
        float speed = 1.0f / animation_.duration;
        animation_.progress += speed * deltaTime_;
        if (animation_.progress > 1.0f) {
            if (animation_.loop) {
                animation_.progress = 0.0f;
            } else {
                animation_.progress = 1.0f;
                animation_.playing = false;
            }
        }
    }
    
    // Clear
    glClearColor(0.05f, 0.05f, 0.08f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    
    // Render 3D scene
    renderScene();
    
    // Render ImGui
    renderImGui();
    
    // Swap buffers
    glfwSwapBuffers(window_);
}

void PoseVisualizer::renderScene() {
    glUseProgram(shaderProgram_);
    
    // Calculate MVP matrix
    float aspect = (float)windowWidth_ / (float)windowHeight_;
    Eigen::Matrix4f view = camera_.getViewMatrix();
    Eigen::Matrix4f proj = camera_.getProjectionMatrix(aspect);
    Eigen::Matrix4f mvp = proj * view;
    
    int mvpLoc = glGetUniformLocation(shaderProgram_, "uMVP");
    int pointSizeLoc = glGetUniformLocation(shaderProgram_, "uPointSize");
    int alphaLoc = glGetUniformLocation(shaderProgram_, "uAlpha");
    
    glUniformMatrix4fv(mvpLoc, 1, GL_FALSE, mvp.data());
    
    // Render in order (back to front for transparency)
    if (settings_.showGrid) renderGrid();
    if (settings_.showAxes) renderAxes();
    if (settings_.showPointCloud) renderPointCloud();
    if (settings_.showTrajectory) renderTrajectory(animation_.progress);
    renderCameraFrustums(animation_.progress);
    if (settings_.showTracks) renderTracks(animation_.progress);
}

void PoseVisualizer::renderPointCloud() {
    if (numPoints_ == 0 && numPointsAfter_ == 0 && pointCorrespondences_.empty()) return;
    
    glUniform1f(glGetUniformLocation(shaderProgram_, "uPointSize"), settings_.pointSize);
    
    // Get the appropriate vertex arrays based on color mode
    std::vector<float>* verticesBefore = (settings_.colorMode == ColorMode::RGB) 
        ? &pointCloudVertices_ : &pointCloudVerticesJet_;
    std::vector<float>* verticesAfter = (settings_.colorMode == ColorMode::RGB) 
        ? &pointCloudVerticesAfter_ : &pointCloudVerticesJetAfter_;
    
    // Handle PointMorph mode - point movement (LiDAR BA) then color change (Visual BA)
    if (settings_.pointCloudMode == PointCloudMode::PointMorph) {
        float t = animation_.progress;
        
        // Phase 1 (0-80%): LiDAR BA - points move from before to after position
        // Phase 2 (80-100%): Visual BA - crossfade to After point cloud with new colors
        
        if (t < 0.8f && !pointCorrespondences_.empty()) {
            // Phase 1: Point movement animation (LiDAR BA effect)
            // Remap t from [0, 0.8] to [0, 1]
            float morphT = t / 0.8f;
            float smoothT = morphT * morphT * (3.0f - 2.0f * morphT);
            
            size_t numCorr = pointCorrespondences_.size();
            for (size_t i = 0; i < numCorr; ++i) {
                const auto& corr = pointCorrespondences_[i];
                
                // Linear interpolation of position
                float x = corr.posBefore.x() * (1.0f - smoothT) + corr.posAfter.x() * smoothT;
                float y = corr.posBefore.y() * (1.0f - smoothT) + corr.posAfter.y() * smoothT;
                float z = corr.posBefore.z() * (1.0f - smoothT) + corr.posAfter.z() * smoothT;
                
                size_t idx = i * 6;
                morphVertices_[idx + 0] = x;
                morphVertices_[idx + 1] = y;
                morphVertices_[idx + 2] = z;
                morphVertices_[idx + 3] = corr.color.x();
                morphVertices_[idx + 4] = corr.color.y();
                morphVertices_[idx + 5] = corr.color.z();
            }
            
            glUniform1f(glGetUniformLocation(shaderProgram_, "uAlpha"), settings_.pointCloudAlpha);
            
            glBindVertexArray(morphVAO_);
            glBindBuffer(GL_ARRAY_BUFFER, morphVBO_);
            glBufferData(GL_ARRAY_BUFFER, morphVertices_.size() * sizeof(float),
                         morphVertices_.data(), GL_DYNAMIC_DRAW);
            
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
            glEnableVertexAttribArray(0);
            glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
            glEnableVertexAttribArray(1);
            
            glDrawArrays(GL_POINTS, 0, (GLsizei)numCorr);
            glBindVertexArray(0);
            return;
        }
        else if (t >= 0.8f && numPointsAfter_ > 0 && !verticesAfter->empty()) {
            // Phase 2: Crossfade to After point cloud (Visual BA effect - new colors)
            // Remap t from [0.8, 1.0] to [0, 1]
            float fadeT = (t - 0.8f) / 0.2f;
            float smoothFade = fadeT * fadeT * (3.0f - 2.0f * fadeT);
            
            // First render the morph result (at final position) with fading alpha
            if (!pointCorrespondences_.empty()) {
                float alphaMorph = settings_.pointCloudAlpha * (1.0f - smoothFade);
                
                // Use final positions (smoothT = 1.0)
                size_t numCorr = pointCorrespondences_.size();
                for (size_t i = 0; i < numCorr; ++i) {
                    const auto& corr = pointCorrespondences_[i];
                    size_t idx = i * 6;
                    morphVertices_[idx + 0] = corr.posAfter.x();
                    morphVertices_[idx + 1] = corr.posAfter.y();
                    morphVertices_[idx + 2] = corr.posAfter.z();
                    morphVertices_[idx + 3] = corr.color.x();
                    morphVertices_[idx + 4] = corr.color.y();
                    morphVertices_[idx + 5] = corr.color.z();
                }
                
                glDepthMask(GL_FALSE);
                glUniform1f(glGetUniformLocation(shaderProgram_, "uAlpha"), alphaMorph);
                
                glBindVertexArray(morphVAO_);
                glBindBuffer(GL_ARRAY_BUFFER, morphVBO_);
                glBufferData(GL_ARRAY_BUFFER, morphVertices_.size() * sizeof(float),
                             morphVertices_.data(), GL_DYNAMIC_DRAW);
                
                glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
                glEnableVertexAttribArray(0);
                glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
                glEnableVertexAttribArray(1);
                
                glDrawArrays(GL_POINTS, 0, (GLsizei)numCorr);
            }
            
            // Then render the After point cloud with increasing alpha
            float alphaAfter = settings_.pointCloudAlpha * smoothFade;
            glUniform1f(glGetUniformLocation(shaderProgram_, "uAlpha"), alphaAfter);
            
            glBindVertexArray(pointCloudAfterVAO_);
            glBindBuffer(GL_ARRAY_BUFFER, pointCloudAfterVBO_);
            glBufferData(GL_ARRAY_BUFFER, verticesAfter->size() * sizeof(float),
                         verticesAfter->data(), GL_DYNAMIC_DRAW);
            
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
            glEnableVertexAttribArray(0);
            glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
            glEnableVertexAttribArray(1);
            
            glDrawArrays(GL_POINTS, 0, (GLsizei)numPointsAfter_);
            
            glDepthMask(GL_TRUE);
            glBindVertexArray(0);
            return;
        }
        // Fallback: if no correspondences, just show after cloud
        else if (numPointsAfter_ > 0 && !verticesAfter->empty()) {
            glUniform1f(glGetUniformLocation(shaderProgram_, "uAlpha"), settings_.pointCloudAlpha);
            
            glBindVertexArray(pointCloudAfterVAO_);
            glBindBuffer(GL_ARRAY_BUFFER, pointCloudAfterVBO_);
            glBufferData(GL_ARRAY_BUFFER, verticesAfter->size() * sizeof(float),
                         verticesAfter->data(), GL_DYNAMIC_DRAW);
            
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
            glEnableVertexAttribArray(0);
            glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
            glEnableVertexAttribArray(1);
            
            glDrawArrays(GL_POINTS, 0, (GLsizei)numPointsAfter_);
            glBindVertexArray(0);
            return;
        }
    }
    
    // Handle Crossfade mode with animation - smooth crossfade between clouds
    if (settings_.pointCloudMode == PointCloudMode::Crossfade && 
        numPoints_ > 0 && numPointsAfter_ > 0 && 
        !verticesBefore->empty() && !verticesAfter->empty()) {
        
        float t = animation_.progress;
        
        // Before cloud: starts at 100%, fades out from 0% to 70%
        // After cloud: starts appearing at 30%, reaches 100% at 100%
        // This creates a nice overlap/crossfade between 30% and 70%
        
        // Calculate alpha for "Before" cloud (fade out: 100% at t=0, 0% at t=0.7)
        float alphaBefore = 0.0f;
        if (t < 0.7f) {
            float tBefore = t / 0.7f;  // Normalize to 0-1 range
            // Smooth step for pleasing fade
            float smoothBefore = tBefore * tBefore * (3.0f - 2.0f * tBefore);
            alphaBefore = settings_.pointCloudAlpha * (1.0f - smoothBefore);
        }
        
        // Calculate alpha for "After" cloud (fade in: 0% at t=0.3, 100% at t=1.0)
        float alphaAfter = 0.0f;
        if (t > 0.3f) {
            float tAfter = (t - 0.3f) / 0.7f;  // Normalize to 0-1 range
            tAfter = std::min(1.0f, tAfter);
            // Smooth step for pleasing fade
            float smoothAfter = tAfter * tAfter * (3.0f - 2.0f * tAfter);
            alphaAfter = settings_.pointCloudAlpha * smoothAfter;
        }
        
        // Disable depth writing during crossfade to prevent occlusion issues
        // Both clouds will be blended together properly
        glDepthMask(GL_FALSE);
        
        // Render "After" cloud FIRST (back layer) with increasing alpha
        if (alphaAfter > 0.001f) {
            glUniform1f(glGetUniformLocation(shaderProgram_, "uAlpha"), alphaAfter);
            
            glBindVertexArray(pointCloudAfterVAO_);
            glBindBuffer(GL_ARRAY_BUFFER, pointCloudAfterVBO_);
            glBufferData(GL_ARRAY_BUFFER, verticesAfter->size() * sizeof(float),
                         verticesAfter->data(), GL_DYNAMIC_DRAW);
            
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
            glEnableVertexAttribArray(0);
            glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
            glEnableVertexAttribArray(1);
            
            glDrawArrays(GL_POINTS, 0, (GLsizei)numPointsAfter_);
        }
        
        // Render "Before" cloud SECOND (front layer) with fading alpha
        if (alphaBefore > 0.001f) {
            glUniform1f(glGetUniformLocation(shaderProgram_, "uAlpha"), alphaBefore);
            
            glBindVertexArray(pointCloudVAO_);
            glBindBuffer(GL_ARRAY_BUFFER, pointCloudVBO_);
            glBufferData(GL_ARRAY_BUFFER, verticesBefore->size() * sizeof(float),
                         verticesBefore->data(), GL_DYNAMIC_DRAW);
            
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
            glEnableVertexAttribArray(0);
            glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
            glEnableVertexAttribArray(1);
            
            glDrawArrays(GL_POINTS, 0, (GLsizei)numPoints_);
        }
        
        // Re-enable depth writing
        glDepthMask(GL_TRUE);
        
        glBindVertexArray(0);
        return;
    }
    
    // Non-interpolate modes: render single cloud
    std::vector<float>* vertices = nullptr;
    size_t numPts = 0;
    
    switch (settings_.pointCloudMode) {
        case PointCloudMode::Before:
            vertices = verticesBefore;
            numPts = numPoints_;
            break;
            
        case PointCloudMode::After:
            if (numPointsAfter_ > 0 && !verticesAfter->empty()) {
                vertices = verticesAfter;
                numPts = numPointsAfter_;
            } else {
                // Fallback to before if after not available
                vertices = verticesBefore;
                numPts = numPoints_;
            }
            break;
            
        case PointCloudMode::Crossfade:
        case PointCloudMode::PointMorph:
            // Fallback when one cloud is missing or no correspondences
            if (numPointsAfter_ > 0 && !verticesAfter->empty()) {
                vertices = verticesAfter;
                numPts = numPointsAfter_;
            } else {
                vertices = verticesBefore;
                numPts = numPoints_;
            }
            break;
    }
    
    if (!vertices || vertices->empty() || numPts == 0) return;
    
    glUniform1f(glGetUniformLocation(shaderProgram_, "uAlpha"), settings_.pointCloudAlpha);
    
    // Upload the selected vertex data to the VBO
    glBindVertexArray(pointCloudVAO_);
    glBindBuffer(GL_ARRAY_BUFFER, pointCloudVBO_);
    glBufferData(GL_ARRAY_BUFFER, vertices->size() * sizeof(float),
                 vertices->data(), GL_DYNAMIC_DRAW);
    
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    
    // Draw all points (no limit)
    glDrawArrays(GL_POINTS, 0, (GLsizei)numPts);
    
    glBindVertexArray(0);
}

void PoseVisualizer::renderCameraFrustum(const CameraPose& pose, float r, float g, float b, float alpha) {
    // Camera frustum vertices in camera coordinates
    float size = settings_.cameraSize;
    float aspectRatio = 4.0f / 3.0f;
    float fovY = 60.0f * (float)M_PI / 180.0f;
    float nearDist = size;
    
    float h = nearDist * std::tan(fovY / 2.0f);
    float w = h * aspectRatio;
    
    // Frustum corners in camera frame (camera looking along -Z in OpenGL convention)
    Eigen::Vector3d corners[5];
    corners[0] = Eigen::Vector3d(0, 0, 0);  // Camera center
    corners[1] = Eigen::Vector3d(-w, -h, -nearDist);  // Bottom-left
    corners[2] = Eigen::Vector3d(w, -h, -nearDist);   // Bottom-right
    corners[3] = Eigen::Vector3d(w, h, -nearDist);    // Top-right
    corners[4] = Eigen::Vector3d(-w, h, -nearDist);   // Top-left
    
    // Transform to world coordinates
    Eigen::Matrix3d Rwc = pose.R.transpose();
    Eigen::Vector3d twc = -Rwc * pose.t;
    
    Eigen::Vector3d worldCorners[5];
    for (int i = 0; i < 5; ++i) {
        worldCorners[i] = Rwc * corners[i] + twc;
    }
    
    // Build line vertices
    std::vector<float> lines;
    auto addLine = [&](int i, int j) {
        lines.push_back((float)worldCorners[i].x());
        lines.push_back((float)worldCorners[i].y());
        lines.push_back((float)worldCorners[i].z());
        lines.push_back(r); lines.push_back(g); lines.push_back(b);
        
        lines.push_back((float)worldCorners[j].x());
        lines.push_back((float)worldCorners[j].y());
        lines.push_back((float)worldCorners[j].z());
        lines.push_back(r); lines.push_back(g); lines.push_back(b);
    };
    
    // Lines from center to corners
    addLine(0, 1); addLine(0, 2); addLine(0, 3); addLine(0, 4);
    // Rectangle
    addLine(1, 2); addLine(2, 3); addLine(3, 4); addLine(4, 1);
    
    // Upload and draw
    glBindVertexArray(lineVAO_);
    glBindBuffer(GL_ARRAY_BUFFER, lineVBO_);
    glBufferData(GL_ARRAY_BUFFER, lines.size() * sizeof(float), lines.data(), GL_DYNAMIC_DRAW);
    
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    
    glUniform1f(glGetUniformLocation(shaderProgram_, "uAlpha"), alpha);
    glLineWidth(2.0f);
    glDrawArrays(GL_LINES, 0, (GLsizei)(lines.size() / 6));
    
    glBindVertexArray(0);
}

void PoseVisualizer::renderCameraFrustums(float t) {
    size_t n = std::min(posesBefore_.size(), posesAfter_.size());
    if (n == 0) return;
    
    // Render "before" cameras (red, semi-transparent)
    if (settings_.showCamerasBefore && t < 1.0f) {
        float alpha = settings_.cameraAlphaBefore * (1.0f - t);
        for (const auto& pose : posesBefore_) {
            renderCameraFrustum(pose, 0.9f, 0.2f, 0.2f, alpha);
        }
    }
    
    // Render interpolated/after cameras (green)
    for (size_t i = 0; i < n; ++i) {
        CameraPose interpPose = interpolatePose(posesBefore_[i], posesAfter_[i], t);
        float green = 0.3f + 0.6f * t;
        float red = 0.3f * (1.0f - t);
        renderCameraFrustum(interpPose, red, green, 0.2f, settings_.cameraAlphaAfter);
    }
}

void PoseVisualizer::renderTrajectory(float t) {
    size_t n = std::min(posesBefore_.size(), posesAfter_.size());
    if (n < 2) return;
    
    std::vector<float> lines;
    
    // Trajectory before (red, faded)
    if (settings_.showCamerasBefore && t < 1.0f) {
        for (size_t i = 0; i < n - 1; ++i) {
            Eigen::Vector3d c1 = posesBefore_[i].getCenter();
            Eigen::Vector3d c2 = posesBefore_[i+1].getCenter();
            
            float alpha = 0.5f * (1.0f - t);
            lines.push_back((float)c1.x()); lines.push_back((float)c1.y()); lines.push_back((float)c1.z());
            lines.push_back(0.9f); lines.push_back(0.2f); lines.push_back(0.2f);
            
            lines.push_back((float)c2.x()); lines.push_back((float)c2.y()); lines.push_back((float)c2.z());
            lines.push_back(0.9f); lines.push_back(0.2f); lines.push_back(0.2f);
        }
    }
    
    // Trajectory interpolated (green)
    for (size_t i = 0; i < n - 1; ++i) {
        CameraPose p1 = interpolatePose(posesBefore_[i], posesAfter_[i], t);
        CameraPose p2 = interpolatePose(posesBefore_[i+1], posesAfter_[i+1], t);
        
        Eigen::Vector3d c1 = p1.getCenter();
        Eigen::Vector3d c2 = p2.getCenter();
        
        float green = 0.4f + 0.5f * t;
        lines.push_back((float)c1.x()); lines.push_back((float)c1.y()); lines.push_back((float)c1.z());
        lines.push_back(0.2f); lines.push_back(green); lines.push_back(0.3f);
        
        lines.push_back((float)c2.x()); lines.push_back((float)c2.y()); lines.push_back((float)c2.z());
        lines.push_back(0.2f); lines.push_back(green); lines.push_back(0.3f);
    }
    
    if (lines.empty()) return;
    
    glBindVertexArray(lineVAO_);
    glBindBuffer(GL_ARRAY_BUFFER, lineVBO_);
    glBufferData(GL_ARRAY_BUFFER, lines.size() * sizeof(float), lines.data(), GL_DYNAMIC_DRAW);
    
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    
    glUniform1f(glGetUniformLocation(shaderProgram_, "uAlpha"), 1.0f);
    glLineWidth(settings_.trajectoryWidth);
    glDrawArrays(GL_LINES, 0, (GLsizei)(lines.size() / 6));
    
    glBindVertexArray(0);
}

void PoseVisualizer::renderGrid() {
    std::vector<float> lines;
    float size = settings_.gridSize;
    float spacing = settings_.gridSpacing;
    
    float gray = 0.25f;
    
    for (float x = -size; x <= size; x += spacing) {
        lines.push_back(x); lines.push_back(-size); lines.push_back(0.0f);
        lines.push_back(gray); lines.push_back(gray); lines.push_back(gray);
        
        lines.push_back(x); lines.push_back(size); lines.push_back(0.0f);
        lines.push_back(gray); lines.push_back(gray); lines.push_back(gray);
    }
    
    for (float y = -size; y <= size; y += spacing) {
        lines.push_back(-size); lines.push_back(y); lines.push_back(0.0f);
        lines.push_back(gray); lines.push_back(gray); lines.push_back(gray);
        
        lines.push_back(size); lines.push_back(y); lines.push_back(0.0f);
        lines.push_back(gray); lines.push_back(gray); lines.push_back(gray);
    }
    
    glBindVertexArray(lineVAO_);
    glBindBuffer(GL_ARRAY_BUFFER, lineVBO_);
    glBufferData(GL_ARRAY_BUFFER, lines.size() * sizeof(float), lines.data(), GL_DYNAMIC_DRAW);
    
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    
    glUniform1f(glGetUniformLocation(shaderProgram_, "uAlpha"), 0.3f);
    glLineWidth(1.0f);
    glDrawArrays(GL_LINES, 0, (GLsizei)(lines.size() / 6));
    
    glBindVertexArray(0);
}

void PoseVisualizer::renderAxes() {
    float len = settings_.axesLength;
    std::vector<float> lines = {
        // X axis (red)
        0, 0, 0, 1, 0, 0,
        len, 0, 0, 1, 0, 0,
        // Y axis (green)
        0, 0, 0, 0, 1, 0,
        0, len, 0, 0, 1, 0,
        // Z axis (blue)
        0, 0, 0, 0, 0, 1,
        0, 0, len, 0, 0, 1,
    };
    
    glBindVertexArray(lineVAO_);
    glBindBuffer(GL_ARRAY_BUFFER, lineVBO_);
    glBufferData(GL_ARRAY_BUFFER, lines.size() * sizeof(float), lines.data(), GL_DYNAMIC_DRAW);
    
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    
    glUniform1f(glGetUniformLocation(shaderProgram_, "uAlpha"), 1.0f);
    glLineWidth(3.0f);
    glDrawArrays(GL_LINES, 0, 6);
    
    glBindVertexArray(0);
}

void PoseVisualizer::renderTracks(float t) {
    if (tracks_.empty()) return;
    
    std::vector<float> points;
    
    for (const auto& tr : tracks_) {
        Eigen::Vector3d pt = interpolatePoint(tr.Xw_before, tr.Xw_after, t);
        
        // Color gradient: yellow -> cyan
        float r = 1.0f - 0.5f * t;
        float g = 0.8f;
        float b = 0.2f + 0.6f * t;
        
        points.push_back((float)pt.x());
        points.push_back((float)pt.y());
        points.push_back((float)pt.z());
        points.push_back(r);
        points.push_back(g);
        points.push_back(b);
    }
    
    glBindVertexArray(lineVAO_);
    glBindBuffer(GL_ARRAY_BUFFER, lineVBO_);
    glBufferData(GL_ARRAY_BUFFER, points.size() * sizeof(float), points.data(), GL_DYNAMIC_DRAW);
    
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    
    glUniform1f(glGetUniformLocation(shaderProgram_, "uPointSize"), settings_.trackPointSize);
    glUniform1f(glGetUniformLocation(shaderProgram_, "uAlpha"), 0.9f);
    glDrawArrays(GL_POINTS, 0, (GLsizei)(points.size() / 6));
    
    glBindVertexArray(0);
}

CameraPose PoseVisualizer::interpolatePose(const CameraPose& a, const CameraPose& b, float t) const {
    CameraPose result;
    
    // SLERP for rotation
    Eigen::Quaterniond qa(a.R);
    Eigen::Quaterniond qb(b.R);
    Eigen::Quaterniond qr = qa.slerp(t, qb);
    result.R = qr.toRotationMatrix();
    
    // Linear interpolation for translation
    result.t = a.t * (1.0 - t) + b.t * t;
    
    return result;
}

Eigen::Vector3d PoseVisualizer::interpolatePoint(const Eigen::Vector3d& a, const Eigen::Vector3d& b, float t) const {
    return a * (1.0 - t) + b * t;
}

// ============================================================================
// ImGui rendering
// ============================================================================

void PoseVisualizer::renderImGui() {
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
    
    renderControlPanel();
    renderStatsPanel();
    
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

void PoseVisualizer::renderControlPanel() {
    ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(280, 420), ImGuiCond_FirstUseEver);
    
    ImGui::Begin("Control Panel", nullptr, ImGuiWindowFlags_NoCollapse);
    
    // Animation section
    if (ImGui::CollapsingHeader("Animation", ImGuiTreeNodeFlags_DefaultOpen)) {
        // Progress bar with gradient
        ImVec2 progressSize(ImGui::GetContentRegionAvail().x, 20);
        ImGui::ProgressBar(animation_.progress, progressSize);
        
        ImGui::SliderFloat("Progress", &animation_.progress, 0.0f, 1.0f);
        
        ImGui::Spacing();
        
        // Play/Pause button
        if (ImGui::Button(animation_.playing ? "  Pause  " : "  Play  ", ImVec2(80, 30))) {
            animation_.playing = !animation_.playing;
        }
        ImGui::SameLine();
        if (ImGui::Button("Reset", ImVec2(60, 30))) {
            animation_.progress = 0.0f;
            animation_.playing = false;
        }
        ImGui::SameLine();
        ImGui::Checkbox("Loop", &animation_.loop);
        
        // Duration slider (3 to 30 seconds)
        ImGui::SliderFloat("Duration", &animation_.duration, 3.0f, 30.0f, "%.1f sec");
        
        // Show remaining time
        if (animation_.playing) {
            float remaining = animation_.duration * (1.0f - animation_.progress);
            ImGui::TextColored(ImVec4(0.7f, 0.9f, 0.7f, 1.0f), "Remaining: %.1f sec", remaining);
        }
    }
    
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();
    
    // Display options
    if (ImGui::CollapsingHeader("Display", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Checkbox("Point Cloud", &settings_.showPointCloud);
        if (settings_.showPointCloud) {
            // Point cloud mode (Before/After/Crossfade/PointMorph)
            ImGui::Text("Cloud Mode:");
            int cloudMode = static_cast<int>(settings_.pointCloudMode);
            ImGui::RadioButton("Before##cloud", &cloudMode, 0); ImGui::SameLine();
            ImGui::RadioButton("After##cloud", &cloudMode, 1);
            ImGui::RadioButton("Crossfade##cloud", &cloudMode, 2); ImGui::SameLine();
            ImGui::RadioButton("Morph##cloud", &cloudMode, 3);
            settings_.pointCloudMode = static_cast<PointCloudMode>(cloudMode);
            if (settings_.pointCloudMode == PointCloudMode::Crossfade) {
                ImGui::TextColored(ImVec4(0.5f, 0.8f, 1.0f, 1.0f), 
                    "  Alpha blend: Before -> After");
            }
            if (settings_.pointCloudMode == PointCloudMode::PointMorph) {
                ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.5f, 1.0f), 
                    "  Points move: Before -> After");
                if (pointCorrespondences_.empty()) {
                    ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), 
                        "  (No correspondences loaded)");
                } else {
                    ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), 
                        "  %zu points", pointCorrespondences_.size());
                }
            }
            
            // Color mode (RGB/JET)
            ImGui::Text("Color Mode:");
            int colorMode = static_cast<int>(settings_.colorMode);
            ImGui::RadioButton("RGB", &colorMode, 0); ImGui::SameLine();
            ImGui::RadioButton("JET (Height)", &colorMode, 1);
            settings_.colorMode = static_cast<ColorMode>(colorMode);
            
            ImGui::SliderFloat("Point Size", &settings_.pointSize, 1.0f, 10.0f);
            ImGui::SliderFloat("Point Alpha", &settings_.pointCloudAlpha, 0.1f, 1.0f);
            ImGui::SliderInt("Downsample", &settings_.pointCloudDownsample, 1, 50);
            
            // Show info about current cloud
            ImGui::Separator();
            const char* modeStr = "Before";
            if (settings_.pointCloudMode == PointCloudMode::After) modeStr = "After";
            else if (settings_.pointCloudMode == PointCloudMode::Crossfade) modeStr = "Crossfade";
            else if (settings_.pointCloudMode == PointCloudMode::PointMorph) modeStr = "Morph";
            const char* colorStr = (settings_.colorMode == ColorMode::RGB) ? "RGB" : "JET";
            ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "Mode: %s (%s)", modeStr, colorStr);
            if (settings_.pointCloudMode == PointCloudMode::Crossfade || 
                settings_.pointCloudMode == PointCloudMode::PointMorph) {
                ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Progress: %.0f%%", animation_.progress * 100.0f);
            }
        }
        
        ImGui::Spacing();
        ImGui::Checkbox("Cameras (Before)", &settings_.showCamerasBefore);
        ImGui::Checkbox("Cameras (After)", &settings_.showCamerasAfter);
        ImGui::SliderFloat("Camera Size", &settings_.cameraSize, 0.1f, 2.0f);
        
        ImGui::Spacing();
        ImGui::Checkbox("Trajectory", &settings_.showTrajectory);
        ImGui::SliderFloat("Line Width", &settings_.trajectoryWidth, 1.0f, 5.0f);
        
        ImGui::Spacing();
        ImGui::Checkbox("Grid", &settings_.showGrid);
        ImGui::Checkbox("Axes", &settings_.showAxes);
        ImGui::Checkbox("Track Points", &settings_.showTracks);
    }
    
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();
    
    // Camera controls
    if (ImGui::CollapsingHeader("Camera")) {
        if (ImGui::Button("Fit to Scene")) {
            fitCameraToScene();
        }
        ImGui::SameLine();
        if (ImGui::Button("Top View")) {
            camera_.azimuth = 0;
            camera_.elevation = 89;
        }
        ImGui::SameLine();
        if (ImGui::Button("Front")) {
            camera_.azimuth = 0;
            camera_.elevation = 0;
        }
        
        ImGui::SliderFloat("Distance", &camera_.distance, 1.0f, 500.0f);
        ImGui::SliderFloat("FOV", &camera_.fov, 20.0f, 90.0f);
    }
    
    ImGui::End();
}

void PoseVisualizer::renderStatsPanel() {
    ImGui::SetNextWindowPos(ImVec2((float)windowWidth_ - 260, 10), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(250, 240), ImGuiCond_FirstUseEver);
    
    ImGui::Begin("Statistics", nullptr, ImGuiWindowFlags_NoCollapse);
    
    ImGui::Text("Cameras: %zu", numCameras_);
    ImGui::Text("Tracks: %zu", numTracks_);
    
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();
    
    // Point cloud info
    ImGui::Text("Point Clouds:");
    ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.6f, 1.0f), "  Before: %zu pts", numPoints_);
    ImGui::TextColored(ImVec4(0.6f, 1.0f, 0.6f, 1.0f), "  After:  %zu pts", numPointsAfter_);
    
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();
    
    // Reprojection error with color coding
    ImGui::Text("Reprojection Error:");
    ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "  Before: %.2f px", meanErrorBefore_);
    ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f), "  After:  %.2f px", meanErrorAfter_);
    
    if (meanErrorBefore_ > 0) {
        double improvement = (meanErrorBefore_ - meanErrorAfter_) / meanErrorBefore_ * 100.0;
        ImVec4 color = improvement > 0 ? ImVec4(0.4f, 1.0f, 0.4f, 1.0f) : ImVec4(1.0f, 0.4f, 0.4f, 1.0f);
        ImGui::TextColored(color, "  Improvement: %.1f%%", improvement);
    }
    
    ImGui::Spacing();
    ImGui::Text("Keep Ratio: %.1f%%", convergenceRatio_ * 100.0);
    
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), "1/2/3: Before/After/Animate");
    ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), "C: RGB/JET | Space: Play");
    
    ImGui::End();
}

// ============================================================================
// Input callbacks
// ============================================================================

void PoseVisualizer::mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
    // Don't process if ImGui wants the mouse
    ImGuiIO& io = ImGui::GetIO();
    if (io.WantCaptureMouse) return;
    
    PoseVisualizer* viz = static_cast<PoseVisualizer*>(glfwGetWindowUserPointer(window));
    if (!viz) return;
    
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        viz->camera_.rotating = (action == GLFW_PRESS);
    }
    if (button == GLFW_MOUSE_BUTTON_RIGHT) {
        viz->camera_.panning = (action == GLFW_PRESS);
    }
    
    glfwGetCursorPos(window, &viz->camera_.lastMouseX, &viz->camera_.lastMouseY);
}

void PoseVisualizer::cursorPosCallback(GLFWwindow* window, double xpos, double ypos) {
    ImGuiIO& io = ImGui::GetIO();
    if (io.WantCaptureMouse) return;
    
    PoseVisualizer* viz = static_cast<PoseVisualizer*>(glfwGetWindowUserPointer(window));
    if (!viz) return;
    
    double dx = xpos - viz->camera_.lastMouseX;
    double dy = ypos - viz->camera_.lastMouseY;
    viz->camera_.lastMouseX = xpos;
    viz->camera_.lastMouseY = ypos;
    
    if (viz->camera_.rotating) {
        viz->camera_.azimuth -= (float)dx * viz->camera_.rotationSpeed;
        viz->camera_.elevation += (float)dy * viz->camera_.rotationSpeed;
        
        // Clamp elevation
        viz->camera_.elevation = std::max(-89.0f, std::min(89.0f, viz->camera_.elevation));
    }
    
    if (viz->camera_.panning) {
        // Pan in view plane
        float azimuthRad = viz->camera_.azimuth * (float)M_PI / 180.0f;
        float elevationRad = viz->camera_.elevation * (float)M_PI / 180.0f;
        
        Eigen::Vector3d right(-std::sin(azimuthRad), std::cos(azimuthRad), 0);
        Eigen::Vector3d up(
            -std::cos(azimuthRad) * std::sin(elevationRad),
            -std::sin(azimuthRad) * std::sin(elevationRad),
            std::cos(elevationRad)
        );
        
        float panScale = viz->camera_.panSpeed * viz->camera_.distance;
        viz->camera_.target -= right * dx * panScale;
        viz->camera_.target += up * dy * panScale;
    }
}

void PoseVisualizer::scrollCallback(GLFWwindow* window, double xoffset, double yoffset) {
    ImGuiIO& io = ImGui::GetIO();
    if (io.WantCaptureMouse) return;
    
    PoseVisualizer* viz = static_cast<PoseVisualizer*>(glfwGetWindowUserPointer(window));
    if (!viz) return;
    
    if (yoffset > 0) {
        viz->camera_.distance /= viz->camera_.zoomSpeed;
    } else if (yoffset < 0) {
        viz->camera_.distance *= viz->camera_.zoomSpeed;
    }
    
    viz->camera_.distance = std::max(0.1f, std::min(1000.0f, viz->camera_.distance));
}

void PoseVisualizer::keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (action != GLFW_PRESS) return;
    
    PoseVisualizer* viz = static_cast<PoseVisualizer*>(glfwGetWindowUserPointer(window));
    if (!viz) return;
    
    switch (key) {
        case GLFW_KEY_ESCAPE:
            glfwSetWindowShouldClose(window, GLFW_TRUE);
            break;
        case GLFW_KEY_SPACE:
            viz->animation_.playing = !viz->animation_.playing;
            break;
        case GLFW_KEY_R:
            viz->fitCameraToScene();
            break;
        case GLFW_KEY_G:
            viz->settings_.showGrid = !viz->settings_.showGrid;
            break;
        case GLFW_KEY_P:
            viz->settings_.showPointCloud = !viz->settings_.showPointCloud;
            break;
        case GLFW_KEY_T:
            viz->settings_.showTrajectory = !viz->settings_.showTrajectory;
            break;
        case GLFW_KEY_1:
            // Switch to "Before" cloud
            viz->settings_.pointCloudMode = PointCloudMode::Before;
            std::cout << "[PoseVisualizer] Cloud mode: Before" << std::endl;
            break;
        case GLFW_KEY_2:
            // Switch to "After" cloud
            viz->settings_.pointCloudMode = PointCloudMode::After;
            std::cout << "[PoseVisualizer] Cloud mode: After" << std::endl;
            break;
        case GLFW_KEY_3:
            // Switch to "Crossfade" mode (alpha blend between Before and After)
            viz->settings_.pointCloudMode = PointCloudMode::Crossfade;
            std::cout << "[PoseVisualizer] Cloud mode: Crossfade (alpha blend)" << std::endl;
            break;
        case GLFW_KEY_4:
            // Switch to "PointMorph" mode (each point moves from before to after)
            viz->settings_.pointCloudMode = PointCloudMode::PointMorph;
            std::cout << "[PoseVisualizer] Cloud mode: PointMorph (points move)" << std::endl;
            break;
        case GLFW_KEY_C:
            // Toggle color mode (RGB / JET)
            if (viz->settings_.colorMode == ColorMode::RGB) {
                viz->settings_.colorMode = ColorMode::IntensityJet;
                std::cout << "[PoseVisualizer] Color mode: JET (Height)" << std::endl;
            } else {
                viz->settings_.colorMode = ColorMode::RGB;
                std::cout << "[PoseVisualizer] Color mode: RGB" << std::endl;
            }
            break;
    }
}

void PoseVisualizer::framebufferSizeCallback(GLFWwindow* window, int width, int height) {
    PoseVisualizer* viz = static_cast<PoseVisualizer*>(glfwGetWindowUserPointer(window));
    if (!viz) return;
    
    viz->windowWidth_ = width;
    viz->windowHeight_ = height;
    glViewport(0, 0, width, height);
}

} // namespace lvba
