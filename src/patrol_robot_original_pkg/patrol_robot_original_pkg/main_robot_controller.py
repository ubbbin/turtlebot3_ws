import rclpy
from rclpy.node import Node

from std_msgs.msg import String, Bool, Empty
from geometry_msgs.msg import Twist
import math
import time

# 키보드 입력을 위한 모듈 추가 (select, tty, termios 사용)
import sys
import select # 표준 입력(stdin)에서 데이터가 사용 가능한지 확인하는 데 사용
import tty    # 터미널 설정 변경을 위해 필요 (setcbreak)
import termios # 터미널 설정 저장/복원에 사용
import threading # KeyboardReader를 별도 스레드에서 실행하기 위해 필요

# 색상 코드 정의
GREEN = '\033[92m'
YELLOW = '\033[93m'
CYAN = '\033[96m'
MAGENTA = '\033[95m'
RED = '\033[91m'
RESET = '\033[0m'

# --- 키보드 입력 핸들러 클래스 (select 기반으로 수정) ---
class KeyboardReader:
    def __init__(self, node_callback):
        self.node_callback = node_callback
        self.running = True
        self.old_settings = None
        self.thread = None

    def _read_key_loop(self):
        # 터미널 설정을 임시로 변경합니다. (엔터 없이 즉시 키 입력 받기)
        # 이 부분에서 termios.error: (25, 'Inappropriate ioctl for device')가 발생할 수 있습니다.
        # 이 경우, WSL 환경의 제한 사항일 가능성이 높습니다.
        try:
            self.old_settings = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno()) # Enter 없이 즉시 키 입력 감지
        except termios.error as e:
            self.node_callback(f"ERROR: termios configuration failed: {e}. Keyboard input may not work as expected.")
            self.running = False # 오류 발생 시 스레드 종료
            return # 오류 발생 시 _read_key_loop 즉시 종료

        try:
            while self.running:
                # sys.stdin이 읽을 준비가 되었는지 0.1초마다 확인 (논블로킹)
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    key = sys.stdin.read(1) # 한 글자 읽기
                    if self.running:
                        self.node_callback(key)
                else:
                    # 읽을 키가 없으면 잠시 대기
                    time.sleep(0.01) # 너무 빠른 루프 방지

        except Exception as e:
            # 예상치 못한 다른 오류가 발생할 경우
            self.node_callback(f"ERROR: KeyboardReader unexpected exception: {e}")
            pass
        finally:
            self._restore_terminal()

    def _restore_terminal(self):
        # 원래 터미널 설정으로 되돌립니다.
        if self.old_settings:
            try:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
                self.old_settings = None
            except termios.error as e:
                # 복원 중에도 오류가 발생할 수 있습니다 (예: 이미 터미널이 닫혔거나).
                self.node_callback(f"ERROR: Failed to restore terminal settings: {e}")


    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._read_key_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        # 스레드가 종료될 시간을 줍니다.
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0) # 1초 대기
        self._restore_terminal()


# --- 메인 로봇 컨트롤러 노드 ---
class MainRobotController(Node):
    # 로봇의 메인 상태를 정의합니다.
    STATES = [
        'INITIALIZING',
        'PATROL',                 # 1. 사각형 순찰 모드
        'OBJECT_ALIGNMENT',       # 2-1. 장애물 감지 후 정렬 및 접근 모드
        'OBJECT_CIRCULATE_CAPTURE', # 2-2. 장애물 주변 원형 주행 및 촬영 모드
        'WAIT_FOR_CLEARANCE',     # 2-3. 사용자 입력 대기 모드
        'RETURN_TO_PATROL',       # 2-4. 순찰 복귀 모드
    ]

    def __init__(self):
        super().__init__('main_robot_controller_node')
        self.get_logger().info(f"{GREEN}MainRobotController Node has been started.{RESET}")

        # --- ROS 2 퍼블리셔 ---
        self.patrol_command_pub = self.create_publisher(String, '/patrol_command', 10)
        self.aligner_command_pub = self.create_publisher(String, '/object_aligner_command', 10)
        self.capture_signal_pub = self.create_publisher(Empty, '/stop_signal', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # --- ROS 2 구독자 ---
        self.object_detection_sub = self.create_subscription(
            Bool,
            '/object_detection_status',
            self.object_detection_callback,
            10
        )
        self.alignment_status_sub = self.create_subscription(
            String,
            '/alignment_status',
            self.alignment_status_callback,
            10
        )
        self.user_clearance_sub = self.create_subscription(
            Empty,
            '/user_clearance',
            self.user_clearance_callback,
            10
        )
        self.get_logger().info(f"{CYAN}'/user_clearance' 토픽 구독 시작 (WAIT_FOR_CLEARANCE 상태용).{RESET}")


        # --- 로봇 상태 변수 ---
        self.robot_state = 'INITIALIZING'
        self.is_object_detected = False
        self.patrol_resumption_point = {'yaw_idx': 0, 'motion_state': 'FORWARD'}
        self._last_state_log_time = self.get_clock().now()

        # --- 원형 주행 (OBJECT_CIRCULATE_CAPTURE) 관련 변수 ---
        self.circulate_radius = 0.5
        self.circulate_angular_speed = 0.3
        self.circulate_linear_speed = self.circulate_angular_speed * self.circulate_radius
        self.circulate_segment_angle = math.pi / 2
        self.current_circulate_segment = 0
        self.circulate_start_yaw = 0.0
        self.circulate_start_time = None
        self.circulate_timer = None

        # --- 키보드 입력 설정 ---
        self.key_1_pressed = False # 키 '1'이 눌렸을 때 설정될 플래그
        self.keyboard_reader = KeyboardReader(self._process_keyboard_input)
        self.keyboard_reader.start() # 키보드 입력 스레드 시작
        self.get_logger().info(f"{CYAN}키보드 입력 리스너 시작됨. '1' 키를 눌러 순찰을 시작하세요.{RESET}")

        # --- 제어 루프 타이머 ---
        self.control_loop_dt = 0.1 # 100ms
        self.timer = self.create_timer(self.control_loop_dt, self.main_control_loop)

        self.get_logger().info(f"{GREEN}MainRobotController 초기화됨. 현재 상태: {self.robot_state}{RESET}")
        self.update_state('INITIALIZING') # 초기 상태 설정 및 로그


    # --- 상태 업데이트 헬퍼 함수 ---
    def update_state(self, new_state):
        if new_state not in self.STATES:
            self.get_logger().error(f"{RED}유효하지 않은 상태 전환 요청: {new_state}{RESET}")
            return

        if self.robot_state != new_state:
            self.get_logger().info(f"{YELLOW}상태 변경: {self.robot_state} -> {new_state}{RESET}")
            self.robot_state = new_state
            self._last_state_log_time = self.get_clock().now() # 상태 변경 시간 기록

        # 특정 상태 진입 시 추가 액션
        if self.robot_state == 'PATROL':
            # 순찰 재개 시 속도 0으로 초기화
            self.send_velocity_command(0.0, 0.0)
            pass # 순찰 시작 명령은 main_control_loop에서 반복적으로 발행
        elif self.robot_state == 'OBJECT_ALIGNMENT':
            self.send_velocity_command(0.0, 0.0) # 정렬 모드 진입 시 정지
        elif self.robot_state == 'OBJECT_CIRCULATE_CAPTURE':
            self.send_velocity_command(0.0, 0.0) # 원형 주행 진입 시 정지
            # 기존 circulate_timer가 있다면 파괴 (중복 생성 방지)
            if self.circulate_timer:
                self.circulate_timer.destroy()
                self.circulate_timer = None
            # 원형 주행 로직은 main_control_loop에서 시작됨
        elif self.robot_state == 'WAIT_FOR_CLEARANCE':
            self.send_velocity_command(0.0, 0.0) # 대기 모드 진입 시 정지
        elif self.robot_state == 'RETURN_TO_PATROL':
            self.send_velocity_command(0.0, 0.0) # 복귀 모드 진입 시 정지


    # KeyboardReader 스레드로부터 키 입력을 처리하는 메서드
    def _process_keyboard_input(self, key_char):
        if key_char == '1':
            self.get_logger().info(f"{CYAN}키보드 '1' 감지됨.{RESET}")
            # 메인 제어 루프에서 처리할 수 있도록 플래그 설정
            self.key_1_pressed = True
        elif key_char == '\x03': # Ctrl+C (터미널에서 Ctrl+C 누를 때 전송되는 문자)
            self.get_logger().info(f"{CYAN}Ctrl+C 감지됨. 노드 종료 준비.{RESET}")
            # 이 키 감지는 rclpy.spin()의 KeyboardInterrupt와 별개로 동작할 수 있습니다.
            # 하지만 노드 종료 시 터미널 설정을 복원하는 데는 유용합니다.
        elif key_char.startswith("ERROR:"): # KeyboardReader 내부 오류 메시지 수신
            self.get_logger().error(f"{RED}KeyboardReader 내부 오류: {key_char}{RESET}")


    # --- 콜백 함수들 ---
    def object_detection_callback(self, msg):
        self.is_object_detected = msg.data

    def alignment_status_callback(self, msg):
        status = msg.data
        if status == "ALIGNMENT_COMPLETE":
            if self.robot_state == 'OBJECT_ALIGNMENT':
                self.get_logger().info(f"{GREEN}객체 정렬 및 접근 완료! 원형 주행 준비.{RESET}")
                self.update_state('OBJECT_CIRCULATE_CAPTURE')
        elif status == "RETURN_COMPLETE":
            if self.robot_state == 'RETURN_TO_PATROL':
                self.get_logger().info(f"{GREEN}순찰 경로 복귀 완료! 순찰 재개.{RESET}")
                self.update_state('PATROL')

    def user_clearance_callback(self, msg):
        if self.robot_state == 'WAIT_FOR_CLEARANCE':
            self.get_logger().info(f"{GREEN}사용자로부터 장애물 제거 신호 수신! 순찰 복귀 시작.{RESET}")
            self.update_state('RETURN_TO_PATROL')


    # --- 메인 제어 루프 ---
    def main_control_loop(self):
        current_time = self.get_clock().now()

        # 각 상태에 따른 로직 수행
        if self.robot_state == 'INITIALIZING':
            self.get_logger().info(f"{YELLOW}시스템 초기화 중... '1' 키 입력 대기.{RESET}")
            # 키보드 리더에 의해 설정된 플래그 확인
            if self.key_1_pressed:
                self.get_logger().info(f"{GREEN}키보드 '1' 입력 감지! 순찰 시작.{RESET}")
                self.update_state('PATROL')
                self.key_1_pressed = False # 플래그 초기화
            else:
                pass # 계속 대기

        elif self.robot_state == 'PATROL':
            # 1. 순찰 로직
            self.patrol_command_pub.publish(String(data="START_PATROL"))
            if self.is_object_detected:
                self.get_logger().info(f"{RED}🔴 장애물 감지! 순찰 중지 및 정렬 모드 진입.{RESET}")
                self.patrol_command_pub.publish(String(data="STOP_PATROL"))
                self.update_state('OBJECT_ALIGNMENT')
            else:
                if (current_time - self._last_state_log_time).nanoseconds / 1e9 >= 5.0:
                    self.get_logger().info(f"{MAGENTA}현재 상태: {self.robot_state} - 순찰 중...{RESET}")
                    self._last_state_log_time = current_time

        elif self.robot_state == 'OBJECT_ALIGNMENT':
            # 2-1. 장애물 정렬 및 접근 로직
            self.aligner_command_pub.publish(String(data="START_ALIGNMENT"))
            if (current_time - self._last_state_log_time).nanoseconds / 1e9 >= 5.0:
                    self.get_logger().info(f"{MAGENTA}현재 상태: {self.robot_state} - 장애물 정렬 및 접근 중...{RESET}")
                    self._last_state_log_time = current_time

        elif self.robot_state == 'OBJECT_CIRCULATE_CAPTURE':
            # 2-2. 장애물 주변 원형 주행 및 촬영 로직
            self.aligner_command_pub.publish(String(data="STOP_ALIGNMENT"))
            self.patrol_command_pub.publish(String(data="STOP_PATROL"))

            if self.circulate_timer is None:
                self.get_logger().info(f"{GREEN}원형 주행 및 촬영 시작!{RESET}")
                self.circulate_start_time = self.get_clock().now()
                self.circulate_start_yaw = 0.0
                self.current_circulate_segment = 0
                self.circulate_timer = self.create_timer(self.control_loop_dt, self.circulate_control_loop)
                # 첫 번째 이미지 즉시 촬영
                self.capture_signal_pub.publish(Empty())
                self.get_logger().info(f"{CYAN}📸 이미지 촬영 요청 (시작점){RESET}")


        elif self.robot_state == 'WAIT_FOR_CLEARANCE':
            # 2-3. 사용자 입력 대기
            self.send_velocity_command(0.0, 0.0) # 로봇 정지
            if (current_time - self._last_state_log_time).nanoseconds / 1e9 >= 5.0:
                user_command_str = "ros2 topic pub /user_clearance std_msgs/msg/Empty '{}'"
                self.get_logger().info(f"{YELLOW}현재 상태: {self.robot_state} - 사용자 입력 대기 중... ('{user_command_str}' 또는 Ctrl+C){RESET}")
                self._last_state_log_time = current_time

        elif self.robot_state == 'RETURN_TO_PATROL':
            # 2-4. 순찰 복귀
            self.aligner_command_pub.publish(String(data="START_RETURN"))
            if (current_time - self._last_state_log_time).nanoseconds / 1e9 >= 5.0:
                    self.get_logger().info(f"{MAGENTA}현재 상태: {self.robot_state} - 순찰 위치로 복귀 중...{RESET}")
                    self._last_state_log_time = current_time

    # --- 원형 주행 로직 (MainRobotController에서 직접 제어) ---
    def circulate_control_loop(self):
        if self.robot_state != 'OBJECT_CIRCULATE_CAPTURE':
            self.get_logger().warn(f"{RED}ERROR: circulate_control_loop이 잘못된 상태에서 실행 중! ({self.robot_state}){RESET}")
            if self.circulate_timer:
                self.circulate_timer.destroy()
                self.circulate_timer = None
            self.send_velocity_command(0.0, 0.0)
            return

        elapsed_time = (self.get_clock().now() - self.circulate_start_time).nanoseconds / 1e9

        target_segment_time = self.circulate_segment_angle / self.circulate_angular_speed

        if elapsed_time < target_segment_time:
            # 원형 주행
            self.send_velocity_command(self.circulate_linear_speed, self.circulate_angular_speed)
        else:
            # 한 세그먼트 완료
            self.send_velocity_command(0.0, 0.0) # 일시 정지
            self.current_circulate_segment += 1
            self.get_logger().info(f"{GREEN}원형 주행 세그먼트 {self.current_circulate_segment} 완료.{RESET}")

            # 이미지 촬영 요청 (세그먼트 완료 시점)
            self.capture_signal_pub.publish(Empty())
            self.get_logger().info(f"{CYAN}📸 이미지 촬영 요청 (세그먼트 {self.current_circulate_segment}){RESET}")

            if self.current_circulate_segment >= 4: # 0, 1, 2, 3 (총 4번)
                self.get_logger().info(f"{GREEN}원형 주행 및 촬영 완료! 사용자 입력 대기.{RESET}")
                if self.circulate_timer:
                    self.circulate_timer.destroy()
                    self.circulate_timer = None
                self.update_state('WAIT_FOR_CLEARANCE')
            else:
                # 다음 세그먼트 시작을 위해 타이머 재설정 (다시 시작 시간 갱신)
                self.circulate_start_time = self.get_clock().now()


    # --- 속도 명령 발행 헬퍼 함수 ---
    def send_velocity_command(self, linear_x, angular_z):
        twist = Twist()
        twist.linear.x = float(linear_x)
        twist.angular.z = float(angular_z)
        self.cmd_vel_pub.publish(twist)

    def destroy_node(self):
        # 노드 종료 시 키보드 리더 스레드를 안전하게 중지
        if hasattr(self, 'keyboard_reader') and self.keyboard_reader:
            self.keyboard_reader.stop()
        super().destroy_node()

# main_robot_controller.py 파일의 가장 아래 main 함수는 그대로 둡니다.
def main(args=None):
    rclpy.init(args=args)
    main_controller_node = MainRobotController()

    try:
        rclpy.spin(main_controller_node)

    except KeyboardInterrupt:
        main_controller_node.get_logger().info('🛑 시스템 종료 요청 (Ctrl+C).')
    finally:
        main_controller_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()