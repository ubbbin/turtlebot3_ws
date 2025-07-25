import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, LaserScan
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import numpy as np
import os
import datetime
from std_msgs.msg import Empty
import time

class LidarCameraFusionNode(Node):
    def __init__(self):
        super().__init__('lidar_camera_fusion_node')
        self.get_logger().info("LIDAR-Camera Fusion Node has been started.")

        self.bridge = CvBridge()

        # --- 카메라 관련 설정 ---
        self.camera_topic = 'camera/image_raw/compressed'
        self.camera_subscription = self.create_subscription(
            CompressedImage,
            self.camera_topic,
            self.camera_callback,
            10)
        self.get_logger().info(f'"{self.camera_topic}" 토픽 구독 시작.')
        self.current_frame = None
        self.image_width = 0
        self.image_height = 0
        self.target_x = 0
        self.object_center_x = None
        self.is_object_detected = False

        # --- LIDAR 관련 설정 ---
        self.lidar_topic = 'scan'
        self.lidar_subscription = self.create_subscription(
            LaserScan,
            self.lidar_topic,
            self.lidar_callback,
            10)
        self.get_logger().info(f'"{self.lidar_topic}" 토픽 구독 시작.')
        self.min_lidar_distance = float('inf') # 전방 최소 거리
        self.obstacle_distance_threshold = 0.4 # m, 이 거리 이내에 장애물 감지 시 회피/정지

        # --- 로봇 제어 설정 ---
        self.publisher_cmd_vel = self.create_publisher(Twist, 'cmd_vel', 10)
        self.control_timer = self.create_timer(0.1, self.control_loop)

        # --- 정렬 및 복귀 관련 설정 ---
        self.kp_angular = 0.005
        # 변경: LIDAR 기반 거리 제어를 위한 kp 값 감소 (접근 속도 줄임)
        self.kp_linear_approach = 0.2  # 기존 0.5에서 0.2로 감소
        self.target_distance_from_object = 0.5 # m, 물체에 접근할 목표 거리 (LIDAR 값)
        self.total_angular_offset = 0.0
        self.last_angular_z = 0.0
        self.last_time = self.get_clock().now()

        # --- 이미지 저장 및 STOP 신호 설정 ---
        self.base_output_dir = os.path.join(os.path.expanduser('~'), "turtlebot_captured_images")
        if not os.path.exists(self.base_output_dir):
            try: os.makedirs(self.base_output_dir)
            except OSError as e: self.get_logger().error(f"디렉토리 생성 오류: {e}")
        self.stop_subscription = self.create_subscription(Empty, '/stop_signal', self.stop_callback, 10)

        self.get_logger().info("LIDAR-Camera Fusion Node has been started. Monitoring for obstacles and red objects.")
        self.get_logger().info(f"LIDAR obstacle detection threshold set to {self.obstacle_distance_threshold} m.")
        self.get_logger().info(f"Target distance from object (LIDAR): {self.target_distance_from_object} m.")


    # --- 카메라 콜백 함수 ---
    def camera_callback(self, msg):
        try:
            cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, 'bgr8')
            self.current_frame = cv_image

            if self.image_width == 0:
                self.image_height, self.image_width, _ = cv_image.shape
                self.target_x = self.image_width // 2

            processed_image, object_center_x, object_area = self.detect_red_object(cv_image)
            self.object_center_x = object_center_x
            self.is_object_detected = (object_center_x is not None)

            cv2.imshow("LIDAR-Camera Fusion Feed", processed_image)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f"Failed to process camera image: {e}")
            self.current_frame = None
            self.is_object_detected = False

    def detect_red_object(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 100, 100])
        upper_red2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)

        kernel = np.ones((5,5),np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        output_image = image.copy()

        largest_area = 0
        target_object_center_x = None
        target_object_area = 0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 500:
                if area > largest_area:
                    largest_area = area
                    x, y, w, h = cv2.boundingRect(cnt)
                    target_object_center_x = x + w // 2

                    cv2.rectangle(output_image, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    cv2.putText(output_image, "Red Obj", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    cv2.circle(output_image, (target_object_center_x, y + h // 2), 5, (0, 255, 255), -1)

        return output_image, target_object_center_x, target_object_area

    # --- LIDAR 콜백 함수 ---
    def lidar_callback(self, msg):
        front_ranges = list(msg.ranges[:45]) + list(msg.ranges[-45:])

        valid_ranges = [r for r in front_ranges if not np.isinf(r) and not np.isnan(r) and r >= msg.range_min and r <= msg.range_max]

        if valid_ranges:
            self.min_lidar_distance = min(valid_ranges)
        else:
            self.min_lidar_distance = float('inf')

    # --- 메인 제어 루프 (타이머 기반) ---
    def control_loop(self):
        twist_msg = Twist()

        current_time = self.get_clock().now()
        dt = (current_time - self.last_time).nanoseconds / 1e9
        self.last_time = current_time

        # 회전 각도 누적 (카메라 정렬 시에만)
        if self.is_object_detected and self.min_lidar_distance > self.obstacle_distance_threshold:
            self.total_angular_offset += self.last_angular_z * dt
        else:
            self.last_angular_z = 0.0

        # --- 로봇 동작 우선순위 로직 ---
        # 1. LIDAR로 전방 장애물 감지 -> 즉시 정지 (최우선)
        if self.min_lidar_distance < self.obstacle_distance_threshold:
            self.get_logger().warn(f"LIDAR 장애물 감지! ({self.min_lidar_distance:.2f} m). 로봇 정지.")
            self.stop_robot()
            return

        # 2. 카메라로 특정 물체 감지 -> 정렬 시도 (각도) 및 LIDAR로 거리 조절 (직선 속도)
        if self.is_object_detected and self.image_width > 0:
            # 카메라로 각도 제어
            error_x = self.target_x - self.object_center_x
            twist_msg.angular.z = self.kp_angular * error_x

            # LIDAR로 직선 속도 제어
            distance_error = self.min_lidar_distance - self.target_distance_from_object
            twist_msg.linear.x = self.kp_linear_approach * distance_error

            max_angular_vel = 0.5
            # 변경: 최대 접근 속도 감소
            max_linear_vel = 0.05 # 기존 0.1에서 0.05로 감소 (5cm/s)

            twist_msg.angular.z = np.clip(twist_msg.angular.z, -max_angular_vel, max_angular_vel)
            twist_msg.linear.x = np.clip(twist_msg.linear.x, -max_linear_vel, max_linear_vel)

            # 정렬 완료 및 거리 도달 시 정지 조건
            angular_aligned = abs(error_x) < 20 # 픽셀 오차
            distance_reached = abs(distance_error) < 0.05 # 5cm 오차

            if angular_aligned and distance_reached:
                twist_msg.angular.z = 0.0
                twist_msg.linear.x = 0.0
                self.get_logger().info(f"물체 정렬 및 거리 도달 완료! 로봇 정지. (LIDAR: {self.min_lidar_distance:.2f}m)")
            else:
                self.get_logger().info(f"정렬 중 - Obj X: {self.object_center_x}, Lidar Dist: {self.min_lidar_distance:.2f}m -> Angular: {twist_msg.angular.z:.2f}, Linear: {twist_msg.linear.x:.2f}, Total Angle: {np.degrees(self.total_angular_offset):.2f} deg")

        # 3. 아무것도 감지 안 되면 로봇 정지
        else:
            twist_msg.linear.x = 0.00
            twist_msg.angular.z = 0.0
            self.get_logger().info("물체/장애물 미감지. 로봇 정지 중...")

        self.publisher_cmd_vel.publish(twist_msg)
        self.last_angular_z = twist_msg.angular.z

    def stop_robot(self):
        stop_twist = Twist()
        stop_twist.linear.x = 0.0
        stop_twist.angular.z = 0.0
        self.publisher_cmd_vel.publish(stop_twist)
        self.last_angular_z = 0.0

    def stop_callback(self, msg):
        self.get_logger().info("STOP 신호 수신! 현재 카메라 프레임을 저장하고 로봇 정지.")
        self.save_current_frame()
        self.stop_robot()
        self.get_logger().info("특정 로직 수행 완료. 원래 방향으로 복귀 시작.")
        self.return_to_original_heading()

    def return_to_original_heading(self):
        self.get_logger().info(f"원래 방향으로 복귀 시작. 총 오프셋 각도: {np.degrees(self.total_angular_offset):.2f}도")
        target_angle = -self.total_angular_offset
        return_angular_speed = 0.3

        if target_angle > 0: angular_vel = return_angular_speed
        else: angular_vel = -return_angular_speed

        twist_msg = Twist()
        start_time = self.get_clock().now()

        rotated_angle_during_return = 0.0

        while rclpy.ok():
            angle_diff = target_angle - rotated_angle_during_return

            current_angular_vel = np.clip(angular_vel * (angle_diff / abs(target_angle) if target_angle != 0 else 1), -return_angular_speed, return_angular_speed)
            if abs(angle_diff) < np.radians(2):
                current_angular_vel = 0.0

            twist_msg.angular.z = current_angular_vel
            self.publisher_cmd_vel.publish(twist_msg)

            if abs(angle_diff) < np.radians(2):
                break

            rclpy.spin_once(self, timeout_sec=0.05)
            rotated_angle_during_return += current_angular_vel * 0.05

            self.get_logger().info(f"복귀 중: 현재 회전 각도 {np.degrees(rotated_angle_during_return):.2f}도 / 목표 {np.degrees(target_angle):.2f}도")


        self.stop_robot()
        self.get_logger().info("원래 방향으로 복귀 완료!")
        self.total_angular_offset = 0.0

    def save_current_frame(self):
        if self.current_frame is not None and self.base_output_dir is not None:
            today_date_str = datetime.datetime.now().strftime("%y-%m-%d")
            date_specific_dir = os.path.join(self.base_output_dir, today_date_str)
            if not os.path.exists(date_specific_dir):
                try: os.makedirs(date_specific_dir)
                except OSError as e: self.get_logger().error(f"디렉토리 생성 오류: {e}")
            timestamp = datetime.datetime.now().strftime("%H-%M-%S")
            filename = os.path.join(date_specific_dir, f"capture_{timestamp}.jpg")
            try:
                cv2.imwrite(filename, self.current_frame)
                self.get_logger().info(f"이미지 저장됨: {filename}")
            except Exception as e: self.get_logger().error(f"이미지 저장 오류: {e}")
        elif self.base_output_dir is None:
            self.get_logger().error("이미지 저장 기본 디렉토리가 유효하지 않습니다.")
        else:
            self.get_logger().warn("저장할 현재 프레임이 없습니다.")


def main(args=None):
    rclpy.init(args=args)
    node = LidarCameraFusionNode()

    print("\n--- LIDAR-카메라 융합 로봇 제어 노드 ---")
    print("로봇은 LIDAR로 40cm 이내 장애물을 감지하면 정지합니다.")
    print("장애물이 없으면 카메라로 빨간색 물체를 찾아 각도 정렬하고, LIDAR로 거리 조절(50cm)을 시도합니다.")
    print("물체/장애물 미감지 시 로봇은 정지합니다.")
    print("'/stop_signal' 토픽 발행 시 현재 화면 이미지 저장 후 정렬 전 방향으로 복귀합니다.")
    print("ROS 2 터미널에서 'ros2 topic pub /stop_signal std_msgs/msg/Empty '{}'' 명령어를 실행하세요.")
    print("노드를 종료하려면 Ctrl+C를 누르십시오.")

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('노드 종료 요청 (Ctrl-C).')
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()

if __name__ == '__main__':
    main()