import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import numpy as np
import os
import datetime
from std_msgs.msg import Empty, Bool, String # Empty, Bool, String 메시지 타입 추가

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

        # --- 새로운 기능: STOP 신호 퍼블리셔 (이미지 저장 요청용) ---
        self.stop_signal_publisher = self.create_publisher(Empty, '/stop_signal', 10)
        self.get_logger().info("'/stop_signal' 토픽 퍼블리셔 생성 (이미지 저장 요청용).")

        # --- 새로운 기능: 객체 감지 상태 퍼블리셔 ---
        self.object_detection_publisher = self.create_publisher(Bool, '/object_detection_status', 10)
        self.get_logger().info("'/object_detection_status' 토픽 퍼블리셔 생성 (객체 감지 여부 알림용).")

        # --- 새로운 기능: 정렬 상태 퍼블리셔 ---
        self.alignment_status_publisher = self.create_publisher(String, '/alignment_status', 10)
        self.get_logger().info("'/alignment_status' 토픽 퍼블리셔 생성 (정렬 완료 및 복귀 완료 알림용).")

        # --- 새로운 기능: 명령 구독자 (중앙 제어 노드로부터) ---
        self.command_subscription = self.create_subscription(
            String,
            '/object_aligner_command', # 중앙 제어 노드로부터 명령을 받을 토픽
            self.command_callback,
            10
        )
        self.get_logger().info("'/object_aligner_command' 토픽 구독 시작.")


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
        self.current_frame = None # 이미지 저장을 위한 프레임 유지

        # --- 정렬 상태 플래그 ---
        self.angular_alignment_threshold = 20
        self.is_angular_aligned = False
        self._object_detected_this_frame = False # 이번 프레임에 객체 감지 여부
        self._last_object_detection_status = False # 이전에 발행된 객체 감지 상태 (중복 발행 방지)

        # --- 복귀 타이머 관련 변수 ---
        self.angular_return_timer = None
        self.linear_return_timer = None
        self.return_start_time = None
        self.return_target_angle = 0.0
        self.return_target_distance = 0.0
        self.return_angular_speed = 0.3 # rad/s
        self.return_linear_speed = 0.05 # m/s

        # --- 새로운 기능: 제어 활성화 플래그 ---
        self._control_active = False # 초기에는 제어 비활성화, 명령을 기다림
        self._current_mode = "IDLE" # "ALIGN", "RETURN", "IDLE"

        self.get_logger().info("터틀봇 객체 정렬 노드 준비 완료.")


    # --- 새로운 기능: 명령 콜백 함수 ---
    def command_callback(self, msg):
        command = msg.data
        if command == "START_ALIGNMENT":
            self.get_logger().info("✅ 정렬 시작 명령 수신!")
            self._control_active = True
            self._current_mode = "ALIGN"
            # 정렬 시작 시 오프셋 초기화
            self.total_angular_offset = 0.0
            self.total_linear_offset = 0.0
            self.is_angular_aligned = False
            self.stop_robot() # 현재 움직임 정지
        elif command == "STOP_ALIGNMENT":
            self.get_logger().info("🛑 정렬 중지 명령 수신!")
            self._control_active = False
            self._current_mode = "IDLE"
            self.stop_robot()
        elif command == "START_RETURN":
            self.get_logger().info("🔄 복귀 시작 명령 수신!")
            self._control_active = True
            self._current_mode = "RETURN"
            self.stop_robot()
            self.start_angular_return() # 복귀 프로세스 시작


    # --- 카메라 이미지 처리 콜백 ---
    def image_callback(self, msg):
        try:
            cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, 'bgr8')
            self.current_frame = cv_image # 이미지 저장 요청 시 사용될 프레임

            if self.image_width == 0:
                self.image_height, self.image_width, _ = cv_image.shape
                self.target_x = self.image_width // 2
                self.get_logger().info(f"이미지 해상도: {self.image_width}x{self.image_height}, 중앙 X: {self.target_x}")

            processed_image, object_center_x, object_area = self.detect_and_draw_roi_and_get_info(cv_image)

            # 객체 감지 상태 발행
            self._object_detected_this_frame = (object_center_x is not None)
            if self._object_detected_this_frame != self._last_object_detection_status:
                status_msg = Bool()
                status_msg.data = self._object_detected_this_frame
                self.object_detection_publisher.publish(status_msg)
                self._last_object_detection_status = self._object_detected_this_frame

            # 로봇 제어 로직 (제어 활성화 상태에서만 실행)
            if self._control_active and self._current_mode == "ALIGN":
                # 복귀 중이 아닐 때만 제어 로직 실행
                if self.angular_return_timer is None and self.linear_return_timer is None:
                    self.control_robot(object_center_x, object_area)
            elif self._control_active and self._current_mode == "RETURN":
                # 복귀 타이머가 알아서 처리
                pass
            else:
                # 제어 비활성화 또는 IDLE 모드일 경우 로봇 정지
                self.stop_robot()


            cv2.imshow("Turtlebot3 Camera Feed with Object Alignment", processed_image)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f"Failed to process image: {e}")
            self.current_frame = None
            self.stop_robot()
            status_msg = Bool()
            status_msg.data = False # 오류 발생 시 객체 미감지 상태 발행
            self.object_detection_publisher.publish(status_msg)
            self._last_object_detection_status = False


    # --- 객체 감지 및 ROI 그리기 함수 ---
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
            if area > 500: # 작은 노이즈 제거
                if area > largest_area:
                    largest_area = area
                    x, y, w, h = cv2.boundingRect(cnt)
                    target_object_center_x = x + w // 2
                    target_object_area = area

                    cv2.rectangle(output_image, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    cv2.putText(output_image, "Red Object", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    cv2.circle(output_image, (target_object_center_x, y + h // 2), 5, (0, 255, 255), -1)

        return output_image, target_object_center_x, target_object_area

    # --- 로봇 제어 함수 ---
    def control_robot(self, object_center_x, object_area):
        twist_msg = Twist()
        current_time = self.get_clock().now()
        dt = (current_time - self.last_time).nanoseconds / 1e9
        self.last_time = current_time

        self.total_angular_offset += self.last_angular_z * dt
        self.total_linear_offset += self.last_linear_x * dt

        area_error = float('inf')

        alignment_done = False # 정렬 완료 여부를 판단하는 플래그

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
                    alignment_done = True # 정렬 및 접근 완료

            if alignment_done:
                self.stop_robot()
                self._control_active = False # 정렬 완료 시 제어 비활성화
                self._current_mode = "IDLE"
                # 중앙 제어 노드에 정렬 완료를 알림
                status_msg = String()
                status_msg.data = "ALIGNMENT_COMPLETE"
                self.alignment_status_publisher.publish(status_msg)
                self.get_logger().info("✅ 객체 정렬 및 접근 완료 신호 발행.")

        else:
            self.get_logger().warn("물체 감지 안 됨 또는 이미지 폭 0. 로봇 정지.")
            self.stop_robot()
            self.is_angular_aligned = False
            self.last_angular_z = 0.0
            self.last_linear_x = 0.0

        self.publisher_cmd_vel.publish(twist_msg)
        self.last_angular_z = twist_msg.angular.z
        self.last_linear_x = twist_msg.linear.x

    # --- 로봇 정지 함수 ---
    def stop_robot(self):
        stop_twist = Twist()
        stop_twist.linear.x = 0.0
        stop_twist.angular.z = 0.0
        self.publisher_cmd_vel.publish(stop_twist)
        self.last_angular_z = 0.0
        self.last_linear_x = 0.0

    # --- 회전 복귀 타이머 콜백 ---
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
            # self.total_angular_offset = 0.0 # 복귀 후 오프셋 초기화는 중앙 노드에서 처리할 수 있도록 유지
            self.start_linear_return()
        else:
            current_angular_vel_to_publish = self.return_angular_speed * np.sign(self.return_target_angle)
            if abs(self.return_target_angle) > 0:
                current_angular_vel_to_publish = np.clip(current_angular_vel_to_publish * (angle_remaining / abs(self.return_target_angle)),
                                                         -self.return_angular_speed, self.return_angular_speed)
            twist_msg.angular.z = current_angular_vel_to_publish
            self.publisher_cmd_vel.publish(twist_msg)

    # --- 선형 복귀 타이머 콜백 ---
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
            # self.total_linear_offset = 0.0 # 복귀 후 오프셋 초기화는 중앙 노드에서 처리할 수 있도록 유지

            # --- 새로운 기능: 복귀 완료 신호 발행 ---
            self._control_active = False # 복귀 완료 시 제어 비활성화
            self._current_mode = "IDLE"
            status_msg = String()
            status_msg.data = "RETURN_COMPLETE"
            self.alignment_status_publisher.publish(status_msg)
            self.get_logger().info("✅ 객체 정렬 복귀 완료 신호 발행.")

        else:
            current_linear_vel_to_publish = self.return_linear_speed * np.sign(self.return_target_distance)
            if abs(self.return_target_distance) > 0:
                current_linear_vel_to_publish = np.clip(current_linear_vel_to_publish * (distance_remaining / abs(self.return_target_distance)),
                                                        -self.return_linear_speed, self.return_linear_speed)
            twist_msg.linear.x = current_linear_vel_to_publish
            self.publisher_cmd_vel.publish(twist_msg)

    # --- 각도 복귀 시작 도우미 함수 ---
    def start_angular_return(self):
        self.return_target_angle = -self.total_angular_offset # 복귀할 각도 (반대 방향)
        self.get_logger().info(f"DEBUG: start_angular_return called. total_angular_offset={np.degrees(self.total_angular_offset):.2f} degrees")


        if abs(self.return_target_angle) < np.radians(2):
            self.get_logger().info("복귀할 각도가 너무 작습니다. 즉시 각도 복귀 완료. 선형 복귀 시작.")
            self.total_angular_offset = 0.0
            self.start_linear_return()
            return

        self.get_logger().info(f"원래 방향으로 복귀 시작. 총 오프셋 각도: {np.degrees(self.total_angular_offset):.2f}도")
        self.return_start_time = self.get_clock().now()
        self.angular_return_timer = self.create_timer(0.05, self.angular_return_timer_callback)

    # --- 선형 복귀 시작 도우미 함수 ---
    def start_linear_return(self):
        self.return_target_distance = -self.total_linear_offset # 복귀할 거리 (반대 방향)
        self.get_logger().info(f"DEBUG: start_linear_return called. total_linear_offset={self.total_linear_offset:.2f} meters")

        if abs(self.return_target_distance) < 0.01:
            self.get_logger().info("복귀할 선형 거리가 너무 작습니다. 즉시 선형 복귀 완료.")
            self.total_linear_offset = 0.0
            # 선형 복귀가 없으므로 바로 복귀 완료 신호 발행
            self._control_active = False
            self._current_mode = "IDLE"
            status_msg = String()
            status_msg.data = "RETURN_COMPLETE"
            self.alignment_status_publisher.publish(status_msg)
            self.get_logger().info("✅ 객체 정렬 복귀 완료 신호 발행 (선형 이동 없음).")
            return

        self.get_logger().info(f"원래 선형 위치로 복귀 시작. 총 오프셋 거리: {self.total_linear_offset:.2f}m")
        self.return_start_time = self.get_clock().now()
        self.linear_return_timer = self.create_timer(0.05, self.linear_return_timer_callback)


def main(args=None):
    rclpy.init(args=args)
    node = TurtlebotObjectAligner()

    print("\n--- 터틀봇 객체 정렬 및 복귀 노드 ---")
    print("이 노드는 중앙 제어 노드의 명령에 따라 객체 정렬 또는 복귀를 수행합니다.")
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