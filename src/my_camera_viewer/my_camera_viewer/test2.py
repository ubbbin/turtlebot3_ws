import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import numpy as np
import os
import datetime
from std_msgs.msg import Empty
import time # 시간 관련 함수 사용을 위해 임포트

class TurtlebotObjectAligner(Node):
    def __init__(self):
        super().__init__('turtlebot_object_aligner_node')
        self.get_logger().info("Turtlebot Object Aligner Node has been started.")

        self.bridge = CvBridge()

        self.camera_topic = 'camera/image_raw/compressed' 
        self.subscription = self.create_subscription(
            CompressedImage,
            self.camera_topic,
            self.image_callback,
            10)
        self.get_logger().info(f'"{self.camera_topic}" 토픽 구독 시작.')

        self.publisher_cmd_vel = self.create_publisher(Twist, 'cmd_vel', 10)
        self.get_logger().info("'cmd_vel' 토픽 퍼블리셔 생성.")

        # --- 추가된 변수 ---
        self.total_angular_offset = 0.0  # 정렬을 위해 회전한 총 각도 (라디안)
        self.last_angular_z = 0.0        # 이전 제어 주기에서의 각속도 값
        self.last_time = self.get_clock().now() # 시간 계산을 위한 마지막 타임스탬프

        # 제어 상수 (조절 필요)
        self.kp_angular = 0.005  
        self.kp_linear = 0.00005 
        self.target_x = 0 
        self.target_object_area = 20000 
        self.image_width = 0
        self.image_height = 0
        self.current_frame = None

        # 이미지 저장 경로
        self.base_output_dir = os.path.join(os.path.expanduser('~'), "turtlebot_captured_images")
        if not os.path.exists(self.base_output_dir):
            try:
                os.makedirs(self.base_output_dir)
                self.get_logger().info(f"기본 저장 디렉토리 '{self.base_output_dir}'를 생성했습니다.")
            except OSError as e:
                self.get_logger().error(f"디렉토리 생성 오류: {self.base_output_dir} - {e}. 권한을 확인하십시오!")
                self.base_output_dir = None

        self.stop_subscription = self.create_subscription(
            Empty,
            '/stop_signal',
            self.stop_callback,
            10
        )
        self.get_logger().info(f"'/stop_signal' 토픽 구독 시작. 이 토픽이 발행되면 이미지가 저장됩니다.")

        self.get_logger().info("터틀봇 객체 정렬 노드가 시작되었습니다.")
        self.get_logger().info("카메라 캡처를 트리거하려면 '/stop_signal' 토픽을 발행하세요 (예: ros2 topic pub /stop_signal std_msgs/msg/Empty '{}').")
        self.get_logger().info("ROS 2 터미널에서 Ctrl+C를 눌러 노드를 종료하십시오.")

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, 'bgr8')
            self.current_frame = cv_image

            if self.image_width == 0:
                self.image_height, self.image_width, _ = cv_image.shape
                self.target_x = self.image_width // 2 
                self.get_logger().info(f"이미지 해상도: {self.image_width}x{self.image_height}, 중앙 X: {self.target_x}")

            processed_image, object_center_x, object_area = self.detect_and_draw_roi_and_get_info(cv_image)

            # 로봇 제어 로직 (회전 각도 기록 포함)
            self.control_robot(object_center_x, object_area)

            cv2.imshow("Turtlebot3 Camera Feed with Object Alignment", processed_image)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f"Failed to process image: {e}")
            self.current_frame = None
            self.stop_robot()

    def detect_and_draw_roi_and_get_info(self, image):
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
                    target_object_area = area

                    cv2.rectangle(output_image, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    cv2.putText(output_image, "Red Object", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    cv2.circle(output_image, (target_object_center_x, y + h // 2), 5, (0, 255, 255), -1)

        return output_image, target_object_center_x, target_object_area

    def control_robot(self, object_center_x, object_area):
        twist_msg = Twist()
        current_time = self.get_clock().now()
        dt = (current_time - self.last_time).nanoseconds / 1e9 # 초 단위 시간 간격
        self.last_time = current_time

        # --- 회전 각도 기록 로직 추가 ---
        # 이전 제어 주기에서의 각속도와 시간 간격을 곱하여 회전량 계산 및 누적
        self.total_angular_offset += self.last_angular_z * dt
        # -------------------------------
        
        if object_center_x is not None and self.image_width > 0:
            error_x = self.target_x - object_center_x 
            twist_msg.angular.z = self.kp_angular * error_x

            area_error = self.target_object_area - object_area
            twist_msg.linear.x = self.kp_linear * area_error

            max_angular_vel = 0.5 
            max_linear_vel = 0.1  
            twist_msg.angular.z = np.clip(twist_msg.angular.z, -max_angular_vel, max_angular_vel)
            twist_msg.linear.x = np.clip(twist_msg.linear.x, -max_linear_vel, max_linear_vel)

            if abs(error_x) < 20 and abs(area_error) < 10000:
                twist_msg.angular.z = 0.0
                twist_msg.linear.x = 0.0
                self.get_logger().info("물체 정렬 및 거리 도달 완료! 로봇 정지.")

            self.get_logger().info(f"Object X: {object_center_x}, Area: {object_area} -> Angular: {twist_msg.angular.z:.2f}, Linear: {twist_msg.linear.x:.2f}, Total Angle: {np.degrees(self.total_angular_offset):.2f} degrees")
        else:
            self.get_logger().warn("물체 감지 안 됨 또는 이미지 폭 0. 로봇 정지.")
            self.stop_robot()
            # 물체가 감지되지 않을 때는 회전 각도 누적을 멈춤
            self.last_angular_z = 0.0 
        
        self.publisher_cmd_vel.publish(twist_msg)
        self.last_angular_z = twist_msg.angular.z # 현재 각속도를 다음 주기를 위해 저장

    def stop_robot(self):
        """로봇을 즉시 정지시키는 함수"""
        stop_twist = Twist()
        stop_twist.linear.x = 0.0
        stop_twist.angular.z = 0.0
        self.publisher_cmd_vel.publish(stop_twist)
        self.last_angular_z = 0.0 # 정지 시 각속도 기록 초기화

    # --- 추가된 함수: 정렬 전 방향으로 복귀 ---
    def return_to_original_heading(self):
        self.get_logger().info(f"원래 방향으로 복귀 시작. 총 오프셋 각도: {np.degrees(self.total_angular_offset):.2f}도")
        
        # 복귀할 각도 (반대 방향)
        target_angle = -self.total_angular_offset 
        current_angle = 0.0 # 복귀 회전 중 누적 각도
        
        # 복귀 회전 속도 (조절 가능)
        return_angular_speed = 0.3 # rad/s

        # 회전 방향 결정
        if target_angle > 0: # 시계 반대 방향 회전 (양수 각도)
            angular_vel = return_angular_speed
        else: # 시계 방향 회전 (음수 각도)
            angular_vel = -return_angular_speed
        
        twist_msg = Twist()
        self.get_logger().info("복귀 회전 중...")

        # 정해진 각도만큼 회전할 때까지 반복
        # 실제 로봇에서는 IMU 등 외부 센서의 각도 정보를 이용하는 것이 더 정확합니다.
        # 여기서는 단순히 시간으로 계산하는 방식이므로 오차가 발생할 수 있습니다.
        # 정확한 복귀를 위해서는 ROS 2의 tf 또는 IMU 데이터 구독이 필요합니다.
        # 이 예시에서는 단순화를 위해 '스핀'을 직접 호출하여 블로킹 방식으로 구현합니다.
        # 실제 시스템에서는 이 함수를 별도의 서비스 콜백 등으로 호출하고, 노드의 메인 스핀과는 분리해야 합니다.

        # 임시 타이머를 사용하여 일정 시간 동안 각속도 발행
        start_time = self.get_clock().now()
        duration_needed = abs(target_angle) / return_angular_speed if return_angular_speed != 0 else 0

        while rclpy.ok() and (self.get_clock().now() - start_time).nanoseconds / 1e9 < duration_needed:
            twist_msg.angular.z = angular_vel
            self.publisher_cmd_vel.publish(twist_msg)
            rclpy.spin_once(self, timeout_sec=0.1) # 짧게 스핀하여 메시지 처리 및 시간 경과

        self.stop_robot() # 회전 완료 후 정지
        self.get_logger().info("원래 방향으로 복귀 완료!")
        self.total_angular_offset = 0.0 # 복귀 후 오프셋 초기화

    def stop_callback(self, msg):
        self.get_logger().info("STOP 신호 수신! 현재 카메라 프레임을 저장하고 로봇 정지.")
        self.save_current_frame()
        self.stop_robot() 

        # --- 특정 로직 수행 후 호출 (예시) ---
        # 여기서 '특정 로직'이 끝났다고 가정하고 복귀 함수를 호출합니다.
        # 실제로는 이 함수가 어떤 서비스 호출이나 다른 ROS 이벤트에 의해 트리거되어야 합니다.
        self.get_logger().info("특정 로직 수행 완료. 원래 방향으로 복귀 시작.")
        self.return_to_original_heading()
        # ------------------------------------

    def save_current_frame(self):
        if self.current_frame is not None and self.base_output_dir is not None:
            today_date_str = datetime.datetime.now().strftime("%y-%m-%d")
            date_specific_dir = os.path.join(self.base_output_dir, today_date_str)
            if not os.path.exists(date_specific_dir):
                try:
                    os.makedirs(date_specific_dir)
                    self.get_logger().info(f"날짜별 디렉토리 '{date_specific_dir}'를 생성했습니다.")
                except OSError as e:
                    self.get_logger().error(f"날짜별 디렉토리 생성 오류: {date_specific_dir} - {e}. 권한을 확인하십시오!")
                    return
            timestamp = datetime.datetime.now().strftime("%H-%M-%S")
            filename = os.path.join(date_specific_dir, f"capture_{timestamp}.jpg")
            try:
                cv2.imwrite(filename, self.current_frame)
                self.get_logger().info(f"이미지 저장됨: {filename}")
            except Exception as e:
                self.get_logger().error(f"이미지 저장 오류: {e}. 저장 경로 권한을 확인하십시오!")
        elif self.base_output_dir is None:
            self.get_logger().error("이미지 저장 기본 디렉토리가 유효하지 않습니다. 초기화 오류를 확인하십시오.")
        else:
            self.get_logger().warn("저장할 현재 프레임이 없습니다. 카메라 메시지를 기다리는 중입니다.")


def main(args=None):
    rclpy.init(args=args)
    node = TurtlebotObjectAligner()
    
    print("\n--- 터틀봇 객체 정렬 및 복귀 노드 ---")
    print("노드가 실행 중입니다. 카메라 영상에 빨간색 물체가 사각형으로 표시되고 로봇이 정렬을 시도합니다.")
    print("'/stop_signal' 토픽이 발행되면 현재 화면의 이미지가 저장되고, '특정 로직' 수행 후 원래 방향으로 복귀합니다.")
    print("ROS 2 터미널에서 'ros2 topic pub /stop_signal std_msgs/msg/Empty '{}'' 명령어를 실행하여 캡처 및 복귀를 트리거하세요.")
    print("노드를 종료하려면 Ctrl+C를 누르십시오.")

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        node.get_logger().info('노드 종료 요청 (Ctrl+C).')
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()

if __name__ == '__main__':
    main()