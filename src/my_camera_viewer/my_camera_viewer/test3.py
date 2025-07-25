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

        # --- 로봇 상태 및 오프셋 변수 ---
        self.total_angular_offset = 0.0  # 정렬을 위해 회전한 총 각도 (라디안)
        self.total_linear_offset = 0.0   # 정렬을 위해 전진/후진한 총 거리 (미터)
        self.last_angular_z = 0.0        # 이전 제어 주기에서의 각속도 값
        self.last_linear_x = 0.0         # 이전 제어 주기에서의 직선 속도 값
        self.last_time = self.get_clock().now() # 시간 계산을 위한 마지막 타임스탬프

        # --- 제어 상수 ---
        self.kp_angular = 0.005  
        self.kp_linear = 0.00005 
        self.target_x = 0 
        self.target_object_area = 20000 
        self.image_width = 0
        self.image_height = 0
        self.current_frame = None

        # --- 정렬 상태 플래그 ---
        self.angular_alignment_threshold = 20 # 픽셀 오차
        self.is_angular_aligned = False # 각도 정렬 완료 플래그
        
        # --- 복귀 타이머 관련 변수 ---
        self.angular_return_timer = None
        self.linear_return_timer = None
        self.return_start_time = None
        self.return_target_angle = 0.0
        self.return_target_distance = 0.0
        self.return_angular_speed = 0.3 # rad/s
        self.return_linear_speed = 0.05 # m/s

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

        self.get_logger().info("터틀봇 객체 정렬 노드가 시작되었습니다.")
        self.get_logger().info("카메라 캡처를 트리거하려면 '/stop_signal' 토픽을 발행하세요 (예: ros2 topic pub /stop_signal std_msgs/msg/Empty '{}').")
        self.get_logger().info("ROS 2 터미널에서 Ctrl+C를 눌러 노드를 종료하십시오.")

    # --- 카메라 이미지 처리 콜백 ---
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
            # 복귀 중이 아닐 때만 제어 로직 실행
            if self.angular_return_timer is None and self.linear_return_timer is None:
                self.control_robot(object_center_x, object_area)

            cv2.imshow("Turtlebot3 Camera Feed with Object Alignment", processed_image)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f"Failed to process image: {e}")
            self.current_frame = None
            self.stop_robot()

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
        dt = (current_time - self.last_time).nanoseconds / 1e9 # 초 단위 시간 간격
        self.last_time = current_time

        # 이전 제어 주기에서의 각속도/직선속도와 시간 간격을 곱하여 회전량/이동량 계산 및 누적
        self.total_angular_offset += self.last_angular_z * dt
        self.total_linear_offset += self.last_linear_x * dt
        
        # area_error 초기화: 'area_error' 참조 에러 방지
        # 초기값으로 float('inf')를 사용하여, 첫 프레임에서는 면적 관련 조건에 해당되지 않도록 합니다.
        area_error = float('inf') 

        if object_center_x is not None and self.image_width > 0:
            error_x = self.target_x - object_center_x 
            
            # 1. 앵귤러 정렬 먼저 수행
            if not self.is_angular_aligned:
                twist_msg.angular.z = self.kp_angular * error_x
                twist_msg.linear.x = 0.0 # 각도 정렬 중에는 직선 움직임 없음

                max_angular_vel = 0.5 
                twist_msg.angular.z = np.clip(twist_msg.angular.z, -max_angular_vel, max_angular_vel)

                if abs(error_x) < self.angular_alignment_threshold:
                    self.is_angular_aligned = True
                    self.get_logger().info("앵귤러 정렬 완료! 이제 리니어 접근 시작.")
                else:
                    self.get_logger().info(f"앵귤러 정렬 중 - Object X: {object_center_x} -> Angular: {twist_msg.angular.z:.2f}, Total Angle: {np.degrees(self.total_angular_offset):.2f} degrees")
            
            # 2. 앵귤러 정렬이 완료되면 리니어 움직임 시작
            else:
                twist_msg.angular.z = 0.0 # 각도 정렬 완료 후에는 각속도 0
                area_error = self.target_object_area - object_area # 여기서 area_error 값 할당
                twist_msg.linear.x = self.kp_linear * area_error

                max_linear_vel = 0.1  
                twist_msg.linear.x = np.clip(twist_msg.linear.x, -max_linear_vel, max_linear_vel)

                self.get_logger().info(f"리니어 접근 중 - Area: {object_area} -> Linear: {twist_msg.linear.x:.2f}")

                if abs(area_error) < 10000: # 면적 오차 허용 범위
                    twist_msg.linear.x = 0.0
                    self.get_logger().info("리니어 접근 완료! 로봇 정지.")
                    self.is_angular_aligned = False # 다음 물체 감지를 위해 플래그 초기화
                
            # 최종 정지 조건 (각도 및 면적 모두 만족)
            # area_error는 이제 항상 정의되어 있으므로 에러가 발생하지 않습니다.
            if self.is_angular_aligned and abs(area_error) < 10000:
                twist_msg.angular.z = 0.0
                twist_msg.linear.x = 0.0
                self.get_logger().info("물체 정렬 및 거리 도달 완료! 로봇 정지.")
                self.is_angular_aligned = False 

        else:
            self.get_logger().warn("물체 감지 안 됨 또는 이미지 폭 0. 로봇 정지.")
            self.stop_robot() # 물체 미감지 시 강제 정지
            self.is_angular_aligned = False # 물체 미감지 시 플래그 초기화
            # 물체가 감지되지 않을 때는 누적을 멈춤
            self.last_angular_z = 0.0 
            self.last_linear_x = 0.0 
        
        self.publisher_cmd_vel.publish(twist_msg)
        self.last_angular_z = twist_msg.angular.z 
        self.last_linear_x = twist_msg.linear.x   

    # --- 로봇 정지 함수 ---
    def stop_robot(self):
        """로봇을 즉시 정지시키는 함수"""
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
        
        # 현재까지 회전한 각도 추정 (시간 * 각속도). 오차가 누적될 수 있음.
        # np.sign(self.return_target_angle)를 곱하여 회전 방향을 유지
        rotated_angle_during_return = self.return_angular_speed * elapsed_time * np.sign(self.return_target_angle)

        angle_remaining = abs(self.return_target_angle) - abs(rotated_angle_during_return)
        
        twist_msg = Twist()
        if angle_remaining <= np.radians(2): # 2도 이내 오차 허용
            twist_msg.angular.z = 0.0
            self.get_logger().info("원래 방향으로 복귀 완료! 다음: 선형 복귀 시작.")
            self.stop_robot()
            self.angular_return_timer.destroy() # 타이머 중지
            self.angular_return_timer = None
            self.total_angular_offset = 0.0 # 복귀 후 오프셋 초기화
            self.start_linear_return() # 선형 복귀 시작 (순서 중요)
        else:
            # 속도 조절 (도착 지점에 가까워질수록 느려지게)
            # 클리핑을 통해 최대 속도를 넘지 않도록 보장
            current_angular_vel_to_publish = self.return_angular_speed * np.sign(self.return_target_angle)
            if abs(self.return_target_angle) > 0:
                current_angular_vel_to_publish = np.clip(current_angular_vel_to_publish * (angle_remaining / abs(self.return_target_angle)), 
                                                         -self.return_angular_speed, self.return_angular_speed)
            twist_msg.angular.z = current_angular_vel_to_publish
            self.publisher_cmd_vel.publish(twist_msg)
            # self.get_logger().info(f"복귀 중: 현재 회전 각도 {np.degrees(rotated_angle_during_return):.2f}도 / 목표 {np.degrees(self.return_target_angle):.2f}도")

    # --- 선형 복귀 타이머 콜백 ---
    def linear_return_timer_callback(self):
        current_time = self.get_clock().now()
        elapsed_time = (current_time - self.return_start_time).nanoseconds / 1e9
        
        # 현재까지 이동한 거리 추정 (시간 * 속도). 오차가 누적될 수 있음.
        # np.sign(self.return_target_distance)를 곱하여 이동 방향을 유지
        current_distance_traveled_during_return = self.return_linear_speed * elapsed_time * np.sign(self.return_target_distance)

        distance_remaining = abs(self.return_target_distance) - abs(current_distance_traveled_during_return)
        
        twist_msg = Twist()
        if distance_remaining <= 0.01: # 1cm 이내 오차 허용
            twist_msg.linear.x = 0.0
            self.get_logger().info("원래 선형 위치로 복귀 완료!")
            self.stop_robot()
            self.linear_return_timer.destroy() # 타이머 중지
            self.linear_return_timer = None
            self.total_linear_offset = 0.0 # 복귀 후 오프셋 초기화
        else:
            # 속도 조절 (도착 지점에 가까워질수록 느려지게)
            # 클리핑을 통해 최대 속도를 넘지 않도록 보장
            current_linear_vel_to_publish = self.return_linear_speed * np.sign(self.return_target_distance)
            if abs(self.return_target_distance) > 0:
                current_linear_vel_to_publish = np.clip(current_linear_vel_to_publish * (distance_remaining / abs(self.return_target_distance)),
                                                        -self.return_linear_speed, self.return_linear_speed)
            twist_msg.linear.x = current_linear_vel_to_publish
            self.publisher_cmd_vel.publish(twist_msg)
            # self.get_logger().info(f"선형 복귀 중: 현재 이동 거리 {current_distance_traveled_during_return:.2f}m / 목표 {self.return_target_distance:.2f}m")

    # --- 복귀 시작 함수 (STOP 신호 수신 시 호출) ---
    def stop_callback(self, msg):
        self.get_logger().info("STOP 신호 수신! 현재 카메라 프레임을 저장하고 로봇 정지.")
        self.save_current_frame()
        self.stop_robot() 

        self.get_logger().info("특정 로직 수행 완료. 복귀 프로세스 시작.")
        self.start_angular_return() # 각도 복귀 먼저 시작

    # --- 각도 복귀 시작 도우미 함수 ---
    def start_angular_return(self):
        self.return_target_angle = -self.total_angular_offset # 복귀할 각도 (반대 방향)
        
        if abs(self.return_target_angle) < np.radians(2): # 2도 미만은 복귀할 필요 없음
            self.get_logger().info("복귀할 각도가 너무 작습니다. 즉시 각도 복귀 완료. 선형 복귀 시작.")
            self.total_angular_offset = 0.0
            self.start_linear_return() # 바로 선형 복귀 시작
            return

        self.get_logger().info(f"원래 방향으로 복귀 시작. 총 오프셋 각도: {np.degrees(self.total_angular_offset):.2f}도")
        self.return_start_time = self.get_clock().now()
        # 0.05초 간격으로 angular_return_timer_callback 호출
        self.angular_return_timer = self.create_timer(0.05, self.angular_return_timer_callback)

    # --- 선형 복귀 시작 도우미 함수 ---
    def start_linear_return(self):
        self.return_target_distance = -self.total_linear_offset # 복귀할 거리 (반대 방향)

        if abs(self.return_target_distance) < 0.01: # 1cm 미만은 복귀할 필요 없음
            self.get_logger().info("복귀할 선형 거리가 너무 작습니다. 즉시 선형 복귀 완료.")
            self.total_linear_offset = 0.0
            return

        self.get_logger().info(f"원래 선형 위치로 복귀 시작. 총 오프셋 거리: {self.total_linear_offset:.2f}m")
        self.return_start_time = self.get_clock().now()
        # 0.05초 간격으로 linear_return_timer_callback 호출
        self.linear_return_timer = self.create_timer(0.05, self.linear_return_timer_callback)

    # --- 현재 프레임 저장 함수 ---
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
    print("  - 먼저 각도 정렬을 완료한 후, 직선으로 접근합니다.")
    print("  - STOP 신호 수신 시, 현재 위치에서 회전 및 전진/후진 거리를 기억하여 원래 방향/위치로 복귀를 시도합니다.")
    print("    **주의: 바퀴 미끄러짐 등으로 인해 선형 복귀에 오차가 발생할 수 있습니다.**") 
    print("    **복귀 중에도 카메라 피드가 원활하게 유지됩니다.**") 
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