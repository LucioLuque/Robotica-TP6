#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <laser_geometry/laser_geometry.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

// Add for std::abs
#include <cmath>

class IcpLocalizer : public rclcpp::Node
{
public:
    IcpLocalizer()
    : Node("icp_localizer"), first_scan_(true)
    {
        // Subscribers
        scan_subscriber_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
            "/scan", 10, std::bind(&IcpLocalizer::scan_callback, this, std::placeholders::_1));

        // Publishers
        pose_publisher_ = this->create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>("/pose_with_covariance", 10);
        current_cloud_publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/current_cloud", 10);
        stable_cloud_publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/stable_cloud", 10);

        // TF Broadcaster
        tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

        // Initialize pose
        current_pose_.header.frame_id = "odom";
        current_pose_.pose.pose.orientation.w = 1.0;
    }

private:
    void scan_callback(const sensor_msgs::msg::LaserScan::SharedPtr msg)
    /// #TODO: Ejecuten ICP cada vez que se reciba un nuevo scan: mantendrán 
    // stable_cloud como la convergencia de la nube de puntos por ICP y actualizarán
    //  current_pose_ y current_cloud_publisher_ con la nueva convergencia
    //  para filtrar odometría y observar el correcto funcionamiento
    /// de la nube de puntos en Rviz2. 
    {
        // Convert LaserScan to PointCloud
        sensor_msgs::msg::PointCloud2 cloud_msg;
        projector_.projectLaser(*msg, cloud_msg);
        cloud_msg.header.frame_id = "base_link";
        
        pcl::PointCloud<pcl::PointXYZ>::Ptr current_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::fromROSMsg(cloud_msg, *current_cloud);

        if (first_scan_)
        {
            stable_cloud_ = current_cloud;
            first_scan_ = false;
            return;
        }

        // Publish clouds for visualization
        sensor_msgs::msg::PointCloud2 stable_cloud_msg;
        pcl::toROSMsg(*stable_cloud_, stable_cloud_msg);
        stable_cloud_msg.header.stamp = this->get_clock()->now();
        stable_cloud_msg.header.frame_id = "odom";
        stable_cloud_publisher_->publish(stable_cloud_msg);

        sensor_msgs::msg::PointCloud2 curr_cloud_msg;
        pcl::toROSMsg(*current_cloud, curr_cloud_msg);
        curr_cloud_msg.header.stamp = this->get_clock()->now();
        curr_cloud_msg.header.frame_id = "odom";
        current_cloud_publisher_->publish(curr_cloud_msg);

        /// #TODO: Ejecuten ICP: roten la nube de puntos y obtengan la respectiva matriz
        //  de transformación, luego usen update_pose para actualizar current_pose_ y 
        // publish_transform para publicar la transformación en Rviz2. NOTA: No olviden filtrar
        /// la convergencia de ICP, si la rotación no converge por debajo de un límite 
        // de rotación, no actualicen current_pose_, de lo contrario divergerá.
        
        // Filtrar puntos NaN
        pcl::PointCloud<pcl::PointXYZ>::Ptr current_filtered(new pcl::PointCloud<pcl::PointXYZ>);
        current_filtered->reserve(current_cloud->size());
        for (const auto &p : current_cloud->points) {
            if (std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z)) {
                current_filtered->push_back(p);
            }
        }

        pcl::PointCloud<pcl::PointXYZ>::Ptr stable_filtered(new pcl::PointCloud<pcl::PointXYZ>);
        stable_filtered->reserve(stable_cloud_->size());
        for (const auto &p : stable_cloud_->points) {
            if (std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z)) {
                stable_filtered->push_back(p);
            }
        }


        Eigen::Matrix4f icp_transform;
        bool converged = icp(current_filtered, stable_filtered, icp_transform);
        if (converged) {
            update_pose(icp_transform);
            publish_transform();
            stable_cloud_ = current_cloud;
        }

         // Con la convergencia aprobada, actualizar la nube estable.
    }

private:
    bool icp(pcl::PointCloud<pcl::PointXYZ>::Ptr source,
                    pcl::PointCloud<pcl::PointXYZ>::Ptr target,
                    Eigen::Matrix4f &final_transform) {
        final_transform = Eigen::Matrix4f::Identity();
        const int MAX_ITERS = 15;
        const float ROT_THRESHOLD = 0.01; // rad ≈ 0.5°
        float angle;

        pcl::PointCloud<pcl::PointXYZ>::Ptr src(new pcl::PointCloud<pcl::PointXYZ>(*source));

        for(int iter = 0; iter < MAX_ITERS; iter++) {
            std::vector<Eigen::Vector2f> src_pts, tgt_pts;
            closest_point_matching(src, target, src_pts, tgt_pts);
            

            //centroides
            Eigen::Vector2f centroid_src = Eigen::Vector2f::Zero();
            Eigen::Vector2f centroid_tgt = Eigen::Vector2f::Zero();
            
            for(size_t i=0;i<src_pts.size();i++) {
                centroid_src += src_pts[i];
                centroid_tgt += tgt_pts[i];
            }
            centroid_src /= src_pts.size();
            centroid_tgt /= tgt_pts.size();
            
            Eigen::Matrix2f H = Eigen::Matrix2f::Zero();
            for(size_t i=0;i<src_pts.size();i++) {
                H += (src_pts[i] - centroid_src) * (tgt_pts[i] - centroid_tgt).transpose();
            }
            
            //SVD
            Eigen::JacobiSVD<Eigen::Matrix2f> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
            Eigen::Matrix2f R = svd.matrixV() * svd.matrixU().transpose();

            if(R.determinant() < 0) {
                Eigen::Matrix2f V = svd.matrixV();
                V.col(1) *= -1;
                R = V * svd.matrixU().transpose();
            }

            Eigen::Vector2f t = centroid_tgt - R * centroid_src;

            //T
            Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
            T(0,0)=R(0,0);
            T(0,1)=R(0,1);
            T(1,0)=R(1,0);
            T(1,1)=R(1,1);
            T(0,3)=t(0);
            T(1,3)=t(1);

            final_transform = T * final_transform;

            for(auto &p: src->points) {
                Eigen::Vector4f pt(p.x,p.y,0,1);
                pt = T * pt;
                p.x = pt(0);
                p.y = pt(1);
            }
            
            //convergencia
            angle = std::atan2(R(1,0), R(0,0));
            if(std::abs(angle) < ROT_THRESHOLD)
                return true;
        }
        return false;
    }


private:
    void closest_point_matching( pcl::PointCloud<pcl::PointXYZ>::Ptr src,
                                 pcl::PointCloud<pcl::PointXYZ>::Ptr target,
                                 std::vector<Eigen::Vector2f> &src_pts,
                                 std::vector<Eigen::Vector2f> &tgt_pts) {
        float min_dist;
        int min_idx;
        float d;
        
        std::vector<bool> used(target->size(), false);
        for(const auto &p : src->points) {
            min_dist = std::numeric_limits<float>::max();
            min_idx = -1;

            for(size_t j = 0; j < target->points.size(); j++) {
                if(used[j]) continue;

                const auto &q = target->points[j];
                d = std::hypot(p.x - q.x, p.y - q.y);

                if(d < min_dist) {
                    min_dist = d;
                    min_idx = j;
                }
            }

            if(min_idx != -1) {
                used[min_idx] = true;
                const auto &q = target->points[min_idx];

                src_pts.push_back(Eigen::Vector2f(p.x,p.y));
                tgt_pts.push_back(Eigen::Vector2f(q.x,q.y));
            }
        }
    }



    


    void update_pose(const Eigen::Matrix4f& transform)
    {
        tf2::Matrix3x3 rot_matrix(
            transform(0, 0), transform(0, 1), transform(0, 2),
            transform(1, 0), transform(1, 1), transform(1, 2),
            transform(2, 0), transform(2, 1), transform(2, 2)
        );
        tf2::Vector3 translation(transform(0, 3), transform(1, 3), transform(2, 3));
        tf2::Transform incremental_transform(rot_matrix, translation);

        tf2::Transform current_transform;
        tf2::fromMsg(current_pose_.pose.pose, current_transform);

        tf2::Transform new_transform = current_transform * incremental_transform;

        tf2::toMsg(new_transform, current_pose_.pose.pose);
    }

    void publish_transform()
    {
        geometry_msgs::msg::TransformStamped t;

        t.header.stamp = this->get_clock()->now();
        t.header.frame_id = "odom";
        t.child_frame_id = "pra_123";

        t.transform.translation.x = current_pose_.pose.pose.position.x;
        t.transform.translation.y = current_pose_.pose.pose.position.y;
        t.transform.translation.z = current_pose_.pose.pose.position.z;
        t.transform.rotation = current_pose_.pose.pose.orientation;
        
        tf_broadcaster_->sendTransform(t);
    }

    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_subscriber_;
    rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr pose_publisher_;
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr current_cloud_publisher_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr stable_cloud_publisher_;

    laser_geometry::LaserProjection projector_;
    
    pcl::PointCloud<pcl::PointXYZ>::Ptr stable_cloud_;
    bool first_scan_;
    geometry_msgs::msg::PoseWithCovarianceStamped current_pose_;
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    
    auto node = std::make_shared<IcpLocalizer>();
    rclcpp::executors::SingleThreadedExecutor executor;
    executor.add_node(node);
    executor.spin();
    
    rclcpp::shutdown();
    return 0;
}
