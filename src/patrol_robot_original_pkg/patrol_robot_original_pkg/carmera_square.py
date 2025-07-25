# camera.py
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
from std_srvs.srv import SetBool # SetBool 서비스 메시지 임포트

# REMOVE: import subprocess # subprocess 모듈 임포트 - This is no longer needed

class TurtlebotObjectAligner(Node):
    def __init__(self):
        super().__init__('turtlebot_object_aligner_node')
        self.get_logger().info("Turtlebot Object Aligner Node has been started.")

        self.bridge = CvBridge()

        # REMOVE: self.square_bot_process = None
        # REMOVE: self.start_square_bot_node() # camera.py will no longer start square_bot.py

        self.camera_topic = 'camera/image_raw/compressed'
        self.subscription = self.create_subscription(
            CompressedImage,
            self.camera_topic,
            self.image_callback,
            10)
        self.get_logger().info(f'"{self.camera_topic}" 토픽 구독 시작.')

        self.publisher_cmd_vel = self.create_publisher(Twist, 'cmd_vel', 10)
        self.get_logger().info("'cmd_vel' 토픽 퍼블리셔 생성.")

        # --- 로봇 상태 및 오프셋 변수 ---
        self.total_angular_offset = 0.0
        self.total_linear_offset = 0.0
        self.last_angular_z = 0.0
        self.last_linear_x = 0.0
        self.last_time = self.get_clock().now()

        # --- 제어 상수 ---
        self.kp_angular = 0.005
        self.kp_linear = 0.00005
        self.target_x = 0
        self.target_object_area = 20000
        self.image_width = 0
        self.image_height = 0
        self.current_frame = None

        # --- 정렬 상태 플래그 ---
        self.angular_alignment_threshold = 20
        self.is_angular_aligned = False
        self.object_detected_and_aligned = False

        # --- 복귀 타이머 관련 변수 ---
        self.angular_return_timer = None
        self.linear_return_timer = None
        self.return_start_time = None
        self.return_target_angle = 0.0
        self.return_target_distance = 0.0
        self.return_angular_speed = 0.3
        self.return_linear_speed = 0.05
        self.is_returning = False

        # --- 이미지 저장 경로 ---
        self.base_output_dir = os.path.join(os.path.expanduser('~'), "turtlebot_captured_images")
        if not os.path.exists(self.base_output_dir):
            try:
                os.makedirs(self.base_output_dir)
                self.get_logger().info(f"기본 저장 디렉토리 '{self.base_output_dir}'를 생성했습니다.")
            except OSError as e:
                self.get_logger().error(f"디렉토리 생성 오류: {self.base_output_dir} - {e}. 권한을 확인하십시오!")
                self.base_output_dir = None

        # --- STOP 신호 구독 ---
        self.stop_subscription = self.create_subscription(
            Empty,
            '/stop_signal',
            self.stop_callback,
            10
        )
        self.get_logger().info(f"'/stop_signal' 토픽 구독 시작. 이 토픽이 발행되면 이미지가 저장됩니다.")

        # --- 2번 노드 제어를 위한 서비스 클라이언트 ---
        self.patrol_control_client = self.create_client(SetBool, '/manual_stop_control')
        # 서비스 대기 시간을 조금 더 길게 설정하여 안정성을 높임
        self.get_logger().info('"/manual_stop_control" 서비스 연결 시도 중...')
        # `wait_for_service`는 서비스가 준비될 때까지 블록킹하므로, square_bot이 먼저 시작되어야 함.
        # 서비스가 준비될 때까지 대기
        while not self.patrol_control_client.wait_for_service(timeout_sec=5.0): # 타임아웃 5초로 증가
            self.get_logger().info('"/manual_stop_control" 서비스 대기 중 (Square Bot이 시작되었는지 확인하세요)...')
        self.get_logger().info('"/manual_stop_control" 서비스 연결 성공.')


        self.get_logger().info("터틀봇 객체 정렬 노드가 시작되었습니다.")
        self.get_logger().info("카메라 캡처를 트리거하려면 '/stop_signal' 토픽을 발행하세요 (예: ros2 topic pub /stop_signal std_msgs/msg/Empty '{}').")
        self.get_logger().info("ROS 2 터미널에서 Ctrl+C를 눌러 노드를 종료하십시오.")

    # REMOVE: start_square_bot_node method is removed as camera.py no longer manages square_bot.py's process.
    # def start_square_bot_node(self):
    #     """
    #     square_bot.py 노드를 별도의 프로세스로 시작합니다.
    #     """
    #     # ... (removed subprocess logic) ...

    # --- 서비스 요청 함수 (동일) ---
    def send_patrol_control_request(self, data_value):
        request = SetBool.Request()
        request.data = data_value
        self.get_logger().info(f"'/manual_stop_control' 서비스 요청 (data: {data_value}) 전송 중...")
        future = self.patrol_control_client.call_async(request)
        future.add_done_callback(self.patrol_control_response_callback)

    def patrol_control_response_callback(self, future):
        try:
            response = future.result()
            if response.success:
                self.get_logger().info(f"'/manual_stop_control' 서비스 응답 성공: {response.message}")
            else:
                self.get_logger().warn(f"'/manual_stop_control' 서비스 응답 실패: {response.message}")
        except Exception as e:
            self.get_logger().error(f"'/manual_stop_control' 서비스 요청 실패: {e}")

    # --- 카메라 이미지 처리 콜백 (동일) ---
    def image_callback(self, msg):
        try:
            cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, 'bgr8')
            self.current_frame = cv_image

            if self.image_width == 0:
                self.image_height, self.image_width, _ = cv_image.shape
                self.target_x = self.image_width // 2
                self.get_logger().info(f"이미지 해상도: {self.image_width}x{self.image_height}, 중앙 X: {self.target_x}")

            processed_image, object_center_x, object_area = self.detect_and_draw_roi_and_get_info(cv_image)

            if not self.is_returning:
                if object_center_x is not None and object_area > 0:
                    if not self.object_detected_and_aligned:
                        self.get_logger().info("🔵 물체 감지! 순찰 노드 중지 요청.")
                        self.send_patrol_control_request(True) # True는 PAUSE
                        self.object_detected_and_aligned = True
                    self.control_robot(object_center_x, object_area)
                else:
                    if self.object_detected_and_aligned:
                        self.get_logger().warn("🔴 물체 감지 실패! 로봇 정지.")
                        self.stop_robot()
                        self.object_detected_and_aligned = False
                        self.is_angular_aligned = False
            cv2.imshow("Turtlebot3 Camera Feed with Object Alignment", processed_image)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f"Failed to process image: {e}")
            self.current_frame = None
            self.stop_robot()

    # --- 로봇 제어 함수 (동일) ---
    def control_robot(self, object_center_x, object_area):
        twist_msg = Twist()
        current_time = self.get_clock().now()
        dt = (current_time - self.last_time).nanoseconds / 1e9
        self.last_time = current_time

        if self.object_detected_and_aligned:
            self.total_angular_offset += self.last_angular_z * dt
            self.total_linear_offset += self.last_linear_x * dt

        if object_center_x is not None and self.image_width > 0:
            error_x = self.target_x - object_center_x

            if not self.is_angular_aligned:
                twist_msg.angular.z = self.kp_angular * error_x
                twist_msg.linear.x = 0.0
                max_angular_vel = 0.5
                twist_msg.angular.z = np.clip(twist_msg.angular.z, -max_angular_vel, max_angular_vel)

                if abs(error_x) < self.angular_alignment_threshold:
                    self.is_angular_aligned = True
                    self.get_logger().info("앵귤러 정렬 완료! 이제 리니어 접근 시작.")
                else:
                    self.get_logger().info(f"앵귤러 정렬 중 - Object X: {object_center_x} -> Angular: {twist_msg.angular.z:.2f}, Total Angle: {np.degrees(self.total_angular_offset):.2f} degrees")
            else:
                twist_msg.angular.z = 0.0
                area_error = self.target_object_area - object_area
                twist_msg.linear.x = self.kp_linear * area_error
                max_linear_vel = 0.1
                twist_msg.linear.x = np.clip(twist_msg.linear.x, -max_linear_vel, max_linear_vel)

                self.get_logger().info(f"리니어 접근 중 - Area: {object_area} -> Linear: {twist_msg.linear.x:.2f}")

                if abs(area_error) < 10000:
                    twist_msg.linear.x = 0.0
                    self.get_logger().info("리니어 접근 완료! 로봇 정지.")
        else:
            self.get_logger().warn("물체 감지 안 됨 또는 이미지 폭 0. 로봇 정지.")
            self.stop_robot()
            self.object_detected_and_aligned = False
            self.is_angular_aligned = False
            self.last_angular_z = 0.0
            self.last_linear_x = 0.0

        self.publisher_cmd_vel.publish(twist_msg)
        self.last_angular_z = twist_msg.angular.z
        self.last_linear_x = twist_msg.linear.x

    # --- 복귀 관련 함수 (동일) ---
    def angular_return_timer_callback(self):
        current_time = self.get_clock().now()
        elapsed_time = (current_time - self.return_start_time).nanoseconds / 1e9

        rotated_angle_during_return = self.return_angular_speed * elapsed_time * np.sign(self.return_target_angle)
        angle_remaining = abs(self.return_target_angle) - abs(rotated_angle_during_return)

        twist_msg = Twist()
        if angle_remaining <= np.radians(2):
            twist_msg.angular.z = 0.0
            self.get_logger().info("원래 방향으로 복귀 완료! 다음: 선형 복귀 시작.")
            self.stop_robot()
            self.angular_return_timer.destroy()
            self.angular_return_timer = None
            self.total_angular_offset = 0.0
            self.start_linear_return()
        else:
            current_angular_vel_to_publish = self.return_angular_speed * np.sign(self.return_target_angle)
            if abs(self.return_target_angle) > 0:
                current_angular_vel_to_publish = np.clip(current_angular_vel_to_publish * (angle_remaining / abs(self.return_target_angle)),
                                                         -self.return_angular_speed, self.return_angular_speed)
            twist_msg.angular.z = current_angular_vel_to_publish
            self.publisher_cmd_vel.publish(twist_msg)

    def linear_return_timer_callback(self):
        current_time = self.get_clock().now()
        elapsed_time = (current_time - self.return_start_time).nanoseconds / 1e9

        current_distance_traveled_during_return = self.return_linear_speed * elapsed_time * np.sign(self.return_target_distance)
        distance_remaining = abs(self.return_target_distance) - abs(current_distance_traveled_during_return)

        twist_msg = Twist()
        if distance_remaining <= 0.01:
            twist_msg.linear.x = 0.0
            self.get_logger().info("원래 선형 위치로 복귀 완료!")
            self.stop_robot()
            self.linear_return_timer.destroy()
            self.linear_return_timer = None
            self.total_linear_offset = 0.0
            self.is_returning = False

            # 복귀 완료 후, 2번 노드에게 순찰 재개 요청 (이 부분이 핵심!)
            self.get_logger().info("🔵 복귀 완료! 순찰 노드 재개 요청.")
            self.send_patrol_control_request(False) # False는 RESUME

        else:
            current_linear_vel_to_publish = self.return_linear_speed * np.sign(self.return_target_distance)
            if abs(self.return_target_distance) > 0:
                current_linear_vel_to_publish = np.clip(current_linear_vel_to_publish * (distance_remaining / abs(self.return_target_distance)),
                                                        -self.return_linear_speed, self.return_linear_speed)
            twist_msg.linear.x = current_linear_vel_to_publish
            self.publisher_cmd_vel.publish(twist_msg)

    def stop_callback(self, msg):
        self.get_logger().info("STOP 신호 수신! 현재 카메라 프레임을 저장하고 로봇 정지.")
        self.save_current_frame()
        self.stop_robot()
        self.is_returning = True
        self.is_angular_aligned = False

        self.get_logger().info("특정 로직 수행 완료. 복귀 프로세스 시작.")
        self.start_angular_return()

    def start_angular_return(self):
        self.return_target_angle = -self.total_angular_offset

        if abs(self.return_target_angle) < np.radians(2):
            self.get_logger().info("복귀할 각도가 너무 작습니다. 즉시 각도 복귀 완료. 선형 복귀 시작.")
            self.total_angular_offset = 0.0
            self.start_linear_return()
            return

        self.get_logger().info(f"원래 방향으로 복귀 시작. 총 오프셋 각도: {np.degrees(self.total_angular_offset):.2f}도")
        self.return_start_time = self.get_clock().now()
        self.angular_return_timer = self.create_timer(0.05, self.angular_return_timer_callback)

    def start_linear_return(self):
        self.return_target_distance = -self.total_linear_offset

        if abs(self.return_target_distance) < 0.01:
            self.get_logger().info("복귀할 선형 거리가 너무 작습니다. 즉시 선형 복귀 완료.")
            self.total_linear_offset = 0.0
            self.is_returning = False
            self.get_logger().info("🔵 복귀 완료! 순찰 노드 재개 요청.")
            self.send_patrol_control_request(False)
            return

        self.get_logger().info(f"원래 선형 위치로 복귀 시작. 총 오프셋 거리: {self.total_linear_offset:.2f}m")
        self.return_start_time = self.get_clock().now()
        self.linear_return_timer = self.create_timer(0.05, self.linear_return_timer_callback)

    # --- 기존 함수들 (detect_and_draw_roi_and_get_info, stop_robot, save_current_frame) ---
    def detect_and_draw_roi_and_get_info(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)
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
                    cv2.putText(output_image, "Black Object", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    cv2.circle(output_image, (target_object_center_x, y + h // 2), 5, (0, 255, 255), -1)
        return output_image, target_object_center_x, target_object_area

    def stop_robot(self):
        stop_twist = Twist()
        stop_twist.linear.x = 0.0
        stop_twist.angular.z = 0.0
        self.publisher_cmd_vel.publish(stop_twist)
        self.last_angular_z = 0.0
        self.last_linear_x = 0.0

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

    def destroy_node(self):
        """노드 종료 시 불필요한 서브프로세스 종료 로직 제거."""
        # REMOVE: if self.square_bot_process: ... (subprocess termination logic removed)
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = TurtlebotObjectAligner()

    print("\n--- 터틀봇 객체 정렬 및 복귀 노드 ---")
    print("이 노드는 'square_patrol_node'가 이미 실행 중이라고 가정합니다.")
    print("  - 'square_patrol_node'를 먼저 별도의 터미널에서 실행해야 합니다.")
    print("  - 물체 감지 시, 'square_patrol_node'는 일시 중지됩니다.")
    print("  - STOP 신호 수신 시, 로봇 복귀 후 'square_patrol_node'는 자동으로 재개됩니다.")
    print("ROS 2 터미널에서 Ctrl+C를 눌러 노드를 종료하십시오.")

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