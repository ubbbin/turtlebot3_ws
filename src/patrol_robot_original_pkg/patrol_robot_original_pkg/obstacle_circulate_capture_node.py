import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Empty, String, Bool # String, Bool 메시지 타입 추가
from sensor_msgs.msg import CompressedImage
import cv2
from cv_bridge import CvBridge
import numpy as np
import os
import datetime
import math

# 색상 코드 정의
GREEN = '\033[92m'
YELLOW = '\033[93m'
CYAN = '\033[96m'
MAGENTA = '\033[95m'
RED = '\033[91m'
RESET = '\033[0m'

# 코드 1: ObstacleCircleAvoider 클래스 (충돌 관련 부분 수정)
class ObstacleCircleAvoider(Node):
    def __init__(self):
        super().__init__('obstacle_circle_avoider')
        self.get_logger().info(f"{GREEN}ObstacleCircleAvoider Node has been started.{RESET}")

        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.scan_sub = self.create_subscription(LaserScan, 'scan', self.scan_callback, 10)
        # 기존 /stop_signal 퍼블리셔는 그대로 유지 (내부 로직에서 사용)
        self.capture_pub = self.create_publisher(Empty, '/stop_signal', 10)
        self.timer = self.create_timer(0.05, self.control_loop)

        # --- 새로운 기능: 외부 제어 명령 구독 ---
        self.avoider_command_sub = self.create_subscription(
            String,
            '/avoider_command', # 중앙 제어 노드로부터 명령을 받을 토픽
            self.avoider_command_callback,
            10
        )
        self.get_logger().info(f"{CYAN}'/avoider_command' 토픽 구독 시작.{RESET}")

        # --- 새로운 기능: 장애물 감지 상태 퍼블리셔 ---
        self.obstacle_avoider_status_pub = self.create_publisher(Bool, '/obstacle_avoider_status', 10)
        self.get_logger().info(f"{CYAN}'/obstacle_avoider_status' 토픽 퍼블리셔 생성.{RESET}")

        self.state = 'move'
        self.start_time = self.get_clock().now()

        self.linear_speed = 0.2
        self.angular_speed = 0.2

        self.closest = float('inf')
        self.circle_step = 0

        self.warned_frame_missing = False
        self.last_capture_time = self.get_clock().now()
        self.stop_log_shown = False

        # --- 새로운 기능: 회피 로직 활성화/비활성화 플래그 ---
        self._avoider_active = False # 초기에는 비활성화, 명령을 기다림
        self._last_obstacle_status_published = False # 중복 발행 방지

    # --- 새로운 기능: 회피 명령 콜백 ---
    def avoider_command_callback(self, msg):
        command = msg.data
        if command == "START_AVOIDER":
            if not self._avoider_active:
                self.get_logger().info(f"{GREEN}✅ 장애물 회피 로직 활성화 명령 수신!{RESET}")
                self._avoider_active = True
                # 회피 로직을 시작 상태로 초기화 (필요하다면)
                self.state = 'move'
                self.start_time = self.get_clock().now()
        elif command == "STOP_AVOIDER":
            if self._avoider_active:
                self.get_logger().info(f"{YELLOW}🛑 장애물 회피 로직 비활성화 명령 수신!{RESET}")
                self._avoider_active = False
                # 즉시 로봇 정지 명령 발행 (다른 노드가 제어권을 가져갈 수 있도록)
                self.cmd_pub.publish(Twist())
                self.state = 'stop' # 내부 상태도 정지로 변경

    def scan_callback(self, msg):
        front_ranges = msg.ranges[0:10] + msg.ranges[-10:]
        self.closest = min(front_ranges)

        # --- 새로운 기능: 장애물 감지 상태 발행 ---
        current_obstacle_detected = False
        if self.closest < 0.5: # 장애물 감지 임계값
            current_obstacle_detected = True

        if current_obstacle_detected != self._last_obstacle_status_published:
            status_msg = Bool()
            status_msg.data = current_obstacle_detected
            self.obstacle_avoider_status_pub.publish(status_msg)
            self._last_obstacle_status_published = current_obstacle_detected
            if current_obstacle_detected:
                self.get_logger().info(f"{RED}🔴 ObstacleCircleAvoider: 전방 장애물 감지! (거리: {self.closest:.2f}m){RESET}")
            else:
                self.get_logger().info(f"{GREEN}🟢 ObstacleCircleAvoider: 전방 장애물 없음.{RESET}")

        # 기존 회피 로직 트리거 (_avoider_active가 True일 때만 내부 상태 변경)
        if self._avoider_active:
            if self.state == 'move' and self.closest < 0.5:
                self.get_logger().info(f"{RED}🛑 ObstacleCircleAvoider: 장애물 감지 → 회피 시작 (거리: {self.closest:.2f}m){RESET}")
                # 이 부분에서 /stop_signal을 발행하지만, 실제 시나리오에서는 MainRobotController가 발행할 예정
                # self.capture_pub.publish(Empty()) # 이 노드의 직접적인 촬영 요청은 MainRobotController가 담당
                self.state = 'turn_right'
                self.start_time = self.get_clock().now()

    def control_loop(self):
        twist = Twist()
        # --- 새로운 기능: _avoider_active가 True일 때만 cmd_vel 발행 ---
        if not self._avoider_active:
            # 비활성화 상태에서는 로봇 정지 명령을 유지
            self.cmd_pub.publish(Twist())
            return

        elapsed = (self.get_clock().now() - self.start_time).nanoseconds / 1e9

        quarter_turn_time = (math.pi / 2) / self.angular_speed
        circle_segment_time = quarter_turn_time + 0.15

        if self.state == 'move':
            twist.linear.x = self.linear_speed

        elif self.state == 'turn_right':
            twist.angular.z = -self.angular_speed
            if elapsed > quarter_turn_time:
                self.state = 'circle_step_0'
                self.start_time = self.get_clock().now()

        elif self.state.startswith('circle_step_'):
            twist.linear.x = self.linear_speed
            twist.angular.z = self.angular_speed
            if elapsed > circle_segment_time:
                if self.circle_step == 3:
                    self.state = 'final_pause_left'
                    self.start_time = self.get_clock().now()
                    self.get_logger().info("🌀 마지막 궤적 완료 → 왼쪽 회전 시작")
                else:
                    self.state = 'pause_left'
                    self.start_time = self.get_clock().now()

        elif self.state == 'pause_left':
            twist.angular.z = self.angular_speed
            if elapsed > quarter_turn_time:
                self.state = 'pause_stop1'
                self.start_time = self.get_clock().now()

        elif self.state == 'pause_stop1':
            twist.angular.z = 0.0
            twist.linear.x = 0.0
            if elapsed > 0.5:
                # 이 부분에서 /stop_signal을 발행하지만, 실제 시나리오에서는 MainRobotController가 발행할 예정
                # self.capture_pub.publish(Empty())
                pass
            if elapsed > 1.5:
                self.state = 'pause_right'
                self.start_time = self.get_clock().now()

        elif self.state == 'pause_right':
            twist.angular.z = -self.angular_speed
            if elapsed > quarter_turn_time:
                self.state = 'pause_stop2'
                self.start_time = self.get_clock().now()

        elif self.state == 'pause_stop2':
            twist.angular.z = 0.0
            twist.linear.x = 0.0
            if elapsed > 0.5:
                # 이 부분에서 /stop_signal을 발행하지만, 실제 시나리오에서는 MainRobotController가 발행할 예정
                # self.capture_pub.publish(Empty())
                pass
            if elapsed > 1.5:
                self.circle_step += 1
                self.state = f'circle_step_{self.circle_step}'
                self.start_time = self.get_clock().now()

        elif self.state == 'final_pause_left':
            twist.angular.z = self.angular_speed
            if elapsed > quarter_turn_time:
                self.state = 'stop'
                self.start_time = self.get_clock().now()
                self.get_logger().info("✅ 마지막 왼쪽 회전 완료 → 정지")

        elif self.state == 'stop':
            twist.angular.z = 0.0
            twist.linear.x = 0.0

        self.cmd_pub.publish(twist)

# 코드 2: TurtlebotCameraCapture 클래스 (이미지 저장 전담 노드로 수정 없음)

class TurtlebotCameraCapture(Node):
    def __init__(self):
        super().__init__('turtlebot_camera_capture')
        self.get_logger().info(f"{GREEN}TurtlebotCameraCapture Node has been started.{RESET}")

        self.camera_topic = '/camera/image_raw/compressed'

        self.sub_image = self.create_subscription(
            CompressedImage,
            self.camera_topic,
            self.image_callback,
            10
        )
        self.get_logger().info(f'"{self.camera_topic}" 토픽 구독 시작.')

        self.cv_bridge = CvBridge()
        self.current_frame = None

        self.base_output_dir = os.path.join(os.path.expanduser('~'), "turtlebot_captured_images")
        if not os.path.exists(self.base_output_dir):
            try:
                os.makedirs(self.base_output_dir)
                self.get_logger().info(f"'{self.base_output_dir}' 디렉토리를 생성했습니다.")
            except OSError as e:
                self.get_logger().error(f"디렉토리 생성 오류: {self.base_output_dir} - {e}. 권한을 확인하십시오!")
                self.base_output_dir = None

        # STOP 토픽 구독자 (이미지 저장 트리거)
        self.stop_subscription = self.create_subscription(
            Empty,
            '/stop_signal',
            self.stop_callback,
            10
        )
        self.get_logger().info(f"{CYAN}'/stop_signal' 토픽 구독 시작. 이 토픽이 발행되면 이미지가 저장됩니다.{RESET}")

        self.last_capture_time = self.get_clock().now() # 캡처 간 최소 시간 간격 제어
        self.capture_cooldown = 0.5 # 캡처 쿨타임 (초)

        self.get_logger().info("터틀봇 카메라 캡처 노드가 준비되었습니다.")
        self.get_logger().info("카메라 캡처를 트리거하려면 '/stop_signal' 토픽을 발행하세요 (예: ros2 topic pub /stop_signal std_msgs/msg/Empty '{}').")
        self.get_logger().info("ROS 2 터미널에서 Ctrl+C를 눌러 노드를 종료하십시오.")


    def image_callback(self, msg):
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            self.current_frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            # (선택 사항) 이미지 처리 및 화면 표시. 실제 로봇에서는 rviz2로 확인하는 것이 일반적
            # cv2.imshow("Turtlebot Camera Feed", self.current_frame)
            # cv2.waitKey(1)
        except Exception as e:
            self.get_logger().error(f"이미지 변환 오류: {e}")
            self.current_frame = None

    def stop_callback(self, msg):
        now = self.get_clock().now()
        # 쿨타임 체크 (너무 짧은 시간 내에 여러 번 캡처 요청이 오면 무시)
        if (now - self.last_capture_time).nanoseconds / 1e9 < self.capture_cooldown:
            # self.get_logger().info("캡처 쿨타임 중. 요청 무시.") # 필요하다면 이 로그 활성화
            return

        self.get_logger().info(f"{YELLOW}STOP 신호 수신! 현재 카메라 프레임을 저장합니다.{RESET}")
        self.save_current_frame()
        self.last_capture_time = now # 캡처 성공 시 시간 업데이트

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
                    return False

            timestamp = datetime.datetime.now().strftime("%H-%M-%S-%f")[:-3] # 밀리초 포함
            filename = os.path.join(date_specific_dir, f"capture_{timestamp}.jpg")

            try:
                cv2.imwrite(filename, self.current_frame)
                self.get_logger().info(f"이미지 저장됨: {filename}")
                return True
            except Exception as e:
                self.get_logger().error(f"이미지 저장 오류: {e}. 저장 경로 권한을 확인하십시오!")
                return False
        elif self.base_output_dir is None:
            self.get_logger().error("이미지 저장 기본 디렉토리가 유효하지 않습니다. 초기화 오류를 확인하십시오.")
            return False
        else:
            self.get_logger().warn("저장할 현재 프레임이 없습니다. 카메라 메시지를 기다리는 중입니다.")
            return False

def main(args=None):
    rclpy.init(args=args)

    # 이 파일은 두 노드를 동시에 실행하기 위해 MultiThreadedExecutor를 사용합니다.
    # 하지만 실제 통합 시스템에서는 main_robot_controller.py가 모든 노드를 관리할 것입니다.
    # 따라서 이 main 함수는 테스트용으로만 유효합니다.
    # 최종 실행 시에는 main_robot_controller.py의 Executor에 이 노드들이 추가됩니다.
    try:
        avoider_node = ObstacleCircleAvoider()
        capture_node = TurtlebotCameraCapture()

        executor = MultiThreadedExecutor()
        executor.add_node(avoider_node)
        executor.add_node(capture_node)

        print("\n--- 통합 노드 실행 (장애물 회피 및 카메라 캡처) ---")
        print("참고: 이 파일은 두 노드를 함께 실행하는 예시입니다. 최종 시스템에서는 'main_robot_controller.py'가 전체를 관리합니다.")
        print("Ctrl+C를 눌러 노드를 종료하십시오.")

        executor.spin()

    except KeyboardInterrupt:
        print('노드 종료 요청 (Ctrl+C).')
    finally:
        if 'executor' in locals() and executor:
            executor.shutdown()
        if 'avoider_node' in locals() and avoider_node:
            avoider_node.destroy_node()
        if 'capture_node' in locals() and capture_node:
            capture_node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()