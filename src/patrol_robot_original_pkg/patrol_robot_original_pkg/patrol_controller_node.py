import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Point
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from tf_transformations import euler_from_quaternion
import math
import time
from std_msgs.msg import String # 순찰 명령 메시지 타입 추가

from rclpy.qos import qos_profile_sensor_data

GREEN = '\033[92m'
YELLOW = '\033[93m'
RESET = '\033[0m'

class PatrolRobotController(Node):
    def __init__(self):
        super().__init__('patrol_robot_controller_node')

        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, qos_profile=qos_profile_sensor_data)

        # --- 새로운 기능: 순찰 명령 구독 ---
        self.patrol_command_sub = self.create_subscription(
            String,
            '/patrol_command', # 중앙 제어 노드로부터 순찰 명령을 받을 토픽
            self.patrol_command_callback,
            10
        )
        self.get_logger().info("'/patrol_command' 토픽 구독 시작.")

        self.control_loop_dt = 0.1
        self.timer = self.create_timer(self.control_loop_dt, self.control_loop)

        self.pose = None
        self.yaw = 0.0

        self.main_state = 'INITIALIZING'
        self.patrol_motion_state = 'IDLE'
        self.current_patrol_idx = 0

        self.patrol_absolute_target_yaws = [
            0.0, 0.0, 0.0, 0.0
        ]
        self._initial_yaw_offset = None

        self.patrol_forward_speed = 0.3
        self.patrol_turn_speed = 0.4
        self.patrol_forward_length = 2.0
        self.patrol_yaw_tolerance = 0.01
        self.patrol_forward_correction_gain = 3.5

        self.patrol_forward_time_target = self.patrol_forward_length / self.patrol_forward_speed
        self.patrol_forward_count_limit = int(self.patrol_forward_time_target / self.control_loop_dt)
        self.patrol_forward_count = 0

        self._odom_initialized = False
        self._scan_received = False
        self._last_warn_time = self.get_clock().now()
        self.last_status_msg = ""

        self.current_linear_x = 0.0
        self.current_angular_z = 0.0
        self.linear_accel_limit = 0.5
        self.angular_accel_limit = 1.0

        # --- 새로운 기능: 순찰 활성화/비활성화 플래그 ---
        self._patrol_active = False # 초기에는 순찰 비활성화, 명령을 기다림

        self.get_logger().info("PatrolRobotController Node has been started.")

    def log_once(self, color, msg):
        if self.last_status_msg != msg:
            self.get_logger().info(f"{color}{msg}{RESET}")
            self.last_status_msg = msg

    def odom_callback(self, msg):
        self.pose = msg.pose.pose.position
        orientation_q = msg.pose.pose.orientation

        _, _, current_absolute_yaw = euler_from_quaternion([
            orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w
        ])
        self.yaw = current_absolute_yaw

        if not self._odom_initialized:
            self._initial_yaw_offset = current_absolute_yaw
            self.patrol_absolute_target_yaws = [
                self.normalize_angle(self._initial_yaw_offset + math.radians(0)),
                self.normalize_angle(self._initial_yaw_offset + math.radians(-90)),
                self.normalize_angle(self._initial_yaw_offset + math.radians(-180)),
                self.normalize_angle(self._initial_yaw_offset + math.radians(90))
            ]
            self.log_once(GREEN, f"🟢 Odom 초기화 완료. 초기 방향: {math.degrees(self._initial_yaw_offset):.2f}도.")
            self.log_once(GREEN, f"🟢 순찰 목표 방향 설정 완료: {[math.degrees(y) for y in self.patrol_absolute_target_yaws]}")
            self._odom_initialized = True
            # 초기화 후 바로 순찰을 시작하는 대신, 명령을 기다리도록 변경
            # self.main_state = 'PATROL'
            # self.patrol_motion_state = 'FORWARD'
            # self.current_patrol_idx = 0
            self.log_once(YELLOW, "⚠️ Odom 초기화 완료. 순찰 시작 명령을 대기합니다.")

    def scan_callback(self, msg):
        if not self._scan_received:
            self._scan_received = True
            self.get_logger().info(f"{YELLOW}⚠️ Lidar Scan 데이터 수신 (순찰 전용 모드에서는 사용 안 함).{RESET}")

    # --- 새로운 기능: 순찰 명령 콜백 ---
    def patrol_command_callback(self, msg):
        command = msg.data
        if command == "START_PATROL":
            if not self._patrol_active:
                self.log_once(GREEN, "✅ 순찰 시작 명령 수신!")
                self._patrol_active = True
                # 순찰 재개 시 필요한 초기화 (예: 현재 위치에서 순찰 시작)
                # Odom 초기화 로직을 다시 실행하여 현재 위치를 순찰 시작점으로 재설정
                if self._odom_initialized: # Odom이 이미 초기화된 경우에만
                    self.main_state = 'PATROL'
                    self.patrol_motion_state = 'FORWARD' # 첫 동작은 직진
                    self.current_patrol_idx = 0 # 첫 번째 목표 방향 (초기 yaw)으로 시작
                    self.patrol_forward_count = 0 # 직진 거리 카운트 초기화
                    self.log_once(GREEN, "🚶 순찰 재개/시작: 직진 모드로 진입.")
                else:
                    self.get_logger().warn(f"{YELLOW}⚠️ Odom 초기화 전 순찰 시작 명령 수신. Odom 대기 중...{RESET}")
        elif command == "STOP_PATROL":
            if self._patrol_active:
                self.log_once(YELLOW, "🛑 순찰 중지 명령 수신!")
                self._patrol_active = False
                self.current_linear_x = 0.0 # 즉시 정지
                self.current_angular_z = 0.0 # 즉시 정지
                self.publisher_.publish(Twist()) # 로봇 정지 명령 발행 (속도 0)
                # 이 시점에서 로봇의 현재 순찰 인덱스와 위치 정보를 유지

        elif command == "RESUME_PATROL": # 추가적인 복귀 후 재개를 위한 명령
            if not self._patrol_active:
                self.log_once(GREEN, "✅ 순찰 재개 명령 수신 (복귀 후)!")
                self._patrol_active = True
                # 복귀가 완료된 후 PatrolRobotController의 순찰을 재개하기 위한 로직.
                # 이때는 기존 순찰 인덱스를 유지하고, 현재 위치에서 다음 순찰 동작을 이어가야 합니다.
                if self._odom_initialized:
                    self.main_state = 'PATROL'
                    # 복귀가 완료되었으니, 다음 사각형 변을 이어가도록 설정
                    # 현재 인덱스와 상태를 그대로 유지하면 됨.
                    self.patrol_motion_state = 'FORWARD' # 복귀 후 바로 직진 시작 가정
                    self.patrol_forward_count = 0 # 복귀 후 새 변의 시작으로 간주하여 카운트 초기화
                    self.log_once(GREEN, f"🚶 순찰 재개: 다음 변으로 이동 시작. (현재 인덱스: {self.current_patrol_idx})")
                else:
                    self.get_logger().warn(f"{YELLOW}⚠️ Odom 초기화 전 순찰 재개 명령 수신. Odom 대기 중...{RESET}")

    def control_loop(self):
        current_time = self.get_clock().now()
        target_linear_x = 0.0
        target_angular_z = 0.0

        # --- 필수 데이터(Odom/Scan) 수신 대기 ---
        if not self._odom_initialized or not self._scan_received:
            if (current_time - self._last_warn_time).nanoseconds / 1e9 >= 5.0:
                self.get_logger().warn(f"{YELLOW}⚠️ 필수 데이터(Odom/Scan) 수신 대기 중... Odom: {self._odom_initialized}, Scan: {self._scan_received}{RESET}")
                self._last_warn_time = current_time
            # 데이터 미수신 시 항상 로봇 정지
            self.current_linear_x = 0.0
            self.current_angular_z = 0.0
            self.publisher_.publish(Twist()) # 데이터 없으면 계속 정지

        # --- 순찰이 활성화된 경우에만 순찰 로직 실행 ---
        elif self._patrol_active:
            if self.main_state == 'INITIALIZING':
                # odom_callback에서 _patrol_active가 True가 되면 PATROL 상태로 전환될 것임
                pass 
            elif self.main_state == 'PATROL':
                self.log_once(GREEN, "🚶 사각형 순찰 중")
                target_yaw = self.patrol_absolute_target_yaws[self.current_patrol_idx]
                yaw_error = self.normalize_angle(target_yaw - self.yaw)

                if self.patrol_motion_state == 'TURN':
                    if abs(yaw_error) > self.patrol_yaw_tolerance:
                        target_angular_z = self.patrol_turn_speed * (yaw_error / abs(yaw_error))
                        target_linear_x = 0.0
                    else:
                        self.patrol_motion_state = 'FORWARD'
                        self.patrol_forward_count = 0
                        target_angular_z = 0.0
                        self.log_once(GREEN, "▶️ 직진 시작")

                elif self.patrol_motion_state == 'FORWARD':
                    if self.patrol_forward_count < self.patrol_forward_count_limit:
                        target_linear_x = self.patrol_forward_speed
                        yaw_error = self.normalize_angle(target_yaw - self.yaw)
                        target_angular_z = self.patrol_forward_correction_gain * yaw_error
                        self.patrol_forward_count += 1
                    else:
                        self.patrol_motion_state = 'TURN'
                        self.current_patrol_idx = (self.current_patrol_idx + 1) % len(self.patrol_absolute_target_yaws)
                        self.log_once(GREEN, f"🏁 한 변 이동 완료. 다음 회전 준비 (다음 목표: {math.degrees(self.patrol_absolute_target_yaws[self.current_patrol_idx]):.2f}도)")
        else:
            # 순찰 비활성화 상태에서는 로봇 정지
            target_linear_x = 0.0
            target_angular_z = 0.0
            self.log_once(YELLOW, "⏸️ 순찰 비활성화. 대기 중...")


        # --- 속도 스무딩 로직 ---
        twist = Twist()

        # 선형 속도 스무딩
        delta_linear_x = target_linear_x - self.current_linear_x
        max_delta_linear = self.linear_accel_limit * self.control_loop_dt

        if abs(delta_linear_x) > max_delta_linear:
            twist.linear.x = self.current_linear_x + (max_delta_linear if delta_linear_x > 0 else -max_delta_linear)
        else:
            twist.linear.x = target_linear_x

        # 각속도 스무딩
        delta_angular_z = target_angular_z - self.current_angular_z
        max_delta_angular = self.angular_accel_limit * self.control_loop_dt

        if abs(delta_angular_z) > max_delta_angular:
            twist.angular.z = self.current_angular_z + (max_delta_angular if delta_angular_z > 0 else -max_delta_angular)
        else:
            twist.angular.z = target_angular_z

        # 다음 제어 주기를 위해 현재 속도 업데이트
        self.current_linear_x = twist.linear.x
        self.current_angular_z = twist.angular.z

        self.publisher_.publish(twist)

    def normalize_angle(self, angle):
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle

def main(args=None):
    rclpy.init(args=args)
    node = PatrolRobotController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('🛑 종료됨 (Ctrl+C)')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()