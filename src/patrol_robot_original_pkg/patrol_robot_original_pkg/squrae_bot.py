# square_bot.py
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Point
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from tf_transformations import euler_from_quaternion
import math
import time
import copy

# ROS 서비스 메시지 임포트
from std_srvs.srv import SetBool

# 색상 코드 정의
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
CYAN = '\033[96m'
MAGENTA = '\033[95m'
RESET = '\033[0m'

class SquarePatrolWithoutObstacleAvoidance(Node):
    def __init__(self):
        super().__init__('square_patrol_node_no_obstacles')

        # ROS 2 Publisher & Subscribers
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        # 참고: LiDAR 구독은 이 코드에서 제거되었습니다. (장애물 회피 로직이 없음)
        # self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, qos_profile=qos_profile_sensor_data)

        # ROS 2 서비스 서버 생성 (수동 정지/재개용)
        self.manual_stop_service = self.create_service(SetBool, '/manual_stop_control', self.handle_manual_stop_request)
        self.get_logger().info(f"{CYAN}💡 ROS Service '/manual_stop_control' (std_srvs/srv/SetBool) 준비 완료. 사용법: ros2 service call /manual_stop_control std_srvs/srv/SetBool 'data: true/false'{RESET}")

        # Control Loop Timer (전체 제어 로직을 0.1초마다 실행)
        self.control_loop_dt = 0.1
        self.timer = self.create_timer(self.control_loop_dt, self.control_loop)

        # Robot State Variables
        self.pose = None
        self.yaw = 0.0

        # --- Main State Machine: 'INITIALIZING', 'PATROL', 'STOPPED' ---
        self.main_state = 'INITIALIZING'
        self.patrol_motion_state = 'IDLE'
        self.current_patrol_idx = 0

        self.patrol_absolute_target_yaws = [
            0.0, 0.0, 0.0, 0.0
        ]
        self._initial_yaw_offset = None

        # Patrol Parameters
        self.patrol_forward_speed = 0.3
        self.patrol_turn_speed = 0.4
        self.patrol_forward_length = 2.0
        self.patrol_yaw_tolerance = 0.01
        self.patrol_forward_correction_gain = 3.0

        # 직진 거리 제어 변수
        self.segment_start_pose = None
        self.segment_start_yaw = 0.0
        self.current_segment_traveled_distance = 0.0
        self.target_segment_length = self.patrol_forward_length

        # --- 수동 정지 관련 변수 ---
        self.manual_stop_requested = False
        self.stopped_reason = ""

        # 정지 시 순찰 상태를 저장할 변수들
        self.saved_patrol_state = {
            'main_state': 'INITIALIZING',
            'patrol_motion_state': 'IDLE',
            'current_patrol_idx': 0,
            'segment_start_pose': None,
            'segment_start_yaw': 0.0,
            'current_segment_traveled_distance': 0.0,
            'target_segment_length': 0.0,
            'stop_pose': None,
            'stop_yaw': 0.0
        }

        # Data Initialization Flags & Logging
        self._odom_initialized = False
        self._last_warn_time = self.get_clock().now()
        self.last_status_msg = ""

        # For smooth acceleration/deceleration
        self.current_linear_x = 0.0
        self.current_angular_z = 0.0
        self.linear_accel_limit = 0.5
        self.angular_accel_limit = 1.0

        # 센서 데이터 수신 타임스탬프 (꺼짐 현상 진단용)
        self._last_odom_time = self.get_clock().now()
        self.sensor_timeout_seconds = 10.0

    def log_once(self, color, msg):
        """이전과 동일한 메시지는 다시 로깅하지 않아 메시지 스팸을 방지합니다."""
        if self.last_status_msg != msg:
            self.get_logger().info(f"{color}{msg}{RESET}")
            self.last_status_msg = msg

    def odom_callback(self, msg):
        """오도메트리 데이터를 수신하여 로봇의 현재 위치와 방향(yaw)을 업데이트하고, 이동 거리를 계산합니다."""
        self._last_odom_time = self.get_clock().now()

        self.pose = msg.pose.pose.position
        orientation_q = msg.pose.pose.orientation

        _, _, current_absolute_yaw = euler_from_quaternion([
            orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w
        ])
        self.yaw = current_absolute_yaw

        if self.main_state == 'PATROL' and self.patrol_motion_state == 'FORWARD' and self.segment_start_pose:
            self.current_segment_traveled_distance = math.sqrt(
                (self.pose.x - self.segment_start_pose.x)**2 +
                (self.pose.y - self.segment_start_pose.y)**2
            )

        if not self._odom_initialized:
            self._initial_yaw_offset = current_absolute_yaw
            # 초기 오프셋을 기준으로 각 사각형 모서리의 절대 각도 계산
            self.patrol_absolute_target_yaws = [
                self.normalize_angle(self._initial_yaw_offset + math.radians(0)),   # 첫 번째 직진 방향 (초기 방향)
                self.normalize_angle(self._initial_yaw_offset + math.radians(-90)),  # 오른쪽으로 90도 회전
                self.normalize_angle(self._initial_yaw_offset + math.radians(-180)), # 또 오른쪽으로 90도 회전 (총 180도)
                self.normalize_angle(self._initial_yaw_offset + math.radians(90))    # 또 오른쪽으로 90도 회전 (총 270도)
            ]
            self.log_once(GREEN, f"🟢 Odom 초기화 완료. 초기 방향: {math.degrees(self._initial_yaw_offset):.2f}도.")
            self.log_once(GREEN, f"🟢 순찰 목표 방향 설정 완료: {[math.degrees(y) for y in self.patrol_absolute_target_yaws]}도")
            self._odom_initialized = True

            self.main_state = 'PATROL'
            self.patrol_motion_state = 'FORWARD' # 처음에는 직진부터 시작
            self.current_patrol_idx = 0 # 첫 번째 변을 순찰

            if self.pose:
                self.segment_start_pose = copy.deepcopy(self.pose)
                self.segment_start_yaw = self.yaw
            self.current_segment_traveled_distance = 0.0
            self.target_segment_length = self.patrol_forward_length
            self.log_once(GREEN, "🚶 초기 회전 없이 바로 직진 순찰 시작. (1번 코너 방향)")


    def handle_manual_stop_request(self, request, response):
        """
        /manual_stop_control 서비스 요청을 처리하여 로봇의 수동 정지 상태를 설정합니다.
        request.data = True: 수동 정지 요청
        request.data = False: 수동 재개 요청
        """
        if request.data: # 수동 정지 요청 (True)
            if not self.manual_stop_requested:
                self.manual_stop_requested = True
                self.log_once(BLUE, "⏸️ 서비스 요청: 수동 정지 활성화. 로봇이 정지합니다.")
                self.stopped_reason = "manual"

                # 정지 시 현재 순찰 상태를 저장
                self.saved_patrol_state['main_state'] = self.main_state
                self.saved_patrol_state['patrol_motion_state'] = self.patrol_motion_state
                self.saved_patrol_state['current_patrol_idx'] = self.current_patrol_idx
                self.saved_patrol_state['stop_pose'] = copy.deepcopy(self.pose)
                self.saved_patrol_state['stop_yaw'] = self.yaw
                self.saved_patrol_state['current_segment_traveled_distance'] = self.current_segment_traveled_distance
                self.saved_patrol_state['target_segment_length'] = self.target_segment_length

                remaining_dist_for_log = self.target_segment_length - self.current_segment_traveled_distance
                self.get_logger().info(f"{BLUE}📦 정지 전 순찰 상태 저장 완료: "
                                        f"코너 {self.current_patrol_idx+1}, "
                                        f"서브상태 '{self.patrol_motion_state}', "
                                        f"남은거리: {remaining_dist_for_log:.2f}m (원래 길이: {self.patrol_forward_length:.2f}m){RESET}")

            response.message = f"수동 정지 요청 받음."
        else: # 수동 재개 요청 (False)
            if self.manual_stop_requested:
                self.manual_stop_requested = False
                self.log_once(BLUE, "▶️ 서비스 요청: 수동 정지 비활성화. 로봇이 재개 조건을 확인합니다.")

                self.stopped_reason = ""

                # 저장된 순찰 상태를 복원하여 재개 준비
                self.current_patrol_idx = self.saved_patrol_state['current_patrol_idx']
                self.patrol_motion_state = self.saved_patrol_state['patrol_motion_state']

                if self.saved_patrol_state['stop_pose']:
                    # 정지된 지점을 새로운 '시작점'으로 간주하여 남은 거리 계산
                    self.segment_start_pose = copy.deepcopy(self.saved_patrol_state['stop_pose'])
                    self.segment_start_yaw = self.saved_patrol_state['stop_yaw']
                    self.current_segment_traveled_distance = 0.0 # 재개 시점부터 다시 0으로 시작

                    remaining_dist = self.saved_patrol_state['target_segment_length'] - self.saved_patrol_state['current_segment_traveled_distance']
                    self.target_segment_length = max(0.0, remaining_dist) # 음수 방지

                    self.log_once(BLUE, f"🔄 순찰 상태 복원 및 재개 지점 설정 완료. "
                                        f"원래 목표 {self.saved_patrol_state['target_segment_length']:.2f}m 중 "
                                        f"{self.saved_patrol_state['current_segment_traveled_distance']:.2f}m 이동, "
                                        f"이제부터 {self.target_segment_length:.2f}m 더 이동합니다.{RESET}")
                else: # 만약 stop_pose가 설정되지 않은 경우 (예: 노드 시작 후 바로 정지)
                    self.segment_start_pose = copy.deepcopy(self.saved_patrol_state['segment_start_pose'])
                    self.segment_start_yaw = self.saved_patrol_state['segment_start_yaw']
                    self.current_segment_traveled_distance = self.saved_patrol_state['current_segment_traveled_distance']
                    self.target_segment_length = self.saved_patrol_state['target_segment_length']

                log_remaining_dist = self.target_segment_length - self.current_segment_traveled_distance
                self.log_once(BLUE, f"🔄 순찰 상태 복원 완료. (코너 {self.current_patrol_idx+1}, 서브상태 '{self.patrol_motion_state}', 최종 남은 이동 거리: {log_remaining_dist:.2f}m){RESET}")

            response.message = f"수동 재개 요청 받음."

        response.success = True
        return response

    def control_loop(self):
        current_time = self.get_clock().now()
        target_linear_x = 0.0
        target_angular_z = 0.0

        time_since_last_odom = (current_time - self._last_odom_time).nanoseconds / 1e9

        if time_since_last_odom > self.sensor_timeout_seconds:
            self.get_logger().error(f"{RED}❌ 치명적 오류: {self.sensor_timeout_seconds}초 이상 Odom 데이터 미수신! Odom: {time_since_last_odom:.2f}s. 노드를 종료합니다.{RESET}")
            raise SystemExit("Odom data timeout, exiting node.")

        if not self._odom_initialized or self.pose is None:
            if (current_time - self._last_warn_time).nanoseconds / 1e9 >= 5.0:
                self.get_logger().warn(f"{YELLOW}⚠️ 필수 데이터(Odom/Pose) 수신 대기 중... Odom 초기화: {self._odom_initialized}, Pose 유효: {self.pose is not None}{RESET}")
                self._last_warn_time = current_time
            self.current_linear_x = 0.0
            self.current_angular_z = 0.0
            self.publisher_.publish(Twist()) # 데이터를 받기 전까지는 로봇을 정지
            return

        prev_main_state = self.main_state

        # --- 수동 정지 로직 (우선 순위 높음) ---
        if self.manual_stop_requested:
            if self.main_state != 'STOPPED': # STOPPED 상태로 진입할 때만 상태 저장
                self.log_once(BLUE, "⏸️ 수동 정지 명령 활성화! 로봇을 정지시킵니다.")
                self.main_state = 'STOPPED'
                self.stopped_reason = "manual"

                # 정지 시 현재 순찰 상태를 저장 (handle_manual_stop_request와 동일한 로직)
                self.saved_patrol_state['main_state'] = self.main_state
                self.saved_patrol_state['patrol_motion_state'] = self.patrol_motion_state
                self.saved_patrol_state['current_patrol_idx'] = self.current_patrol_idx
                self.saved_patrol_state['stop_pose'] = copy.deepcopy(self.pose)
                self.saved_patrol_state['stop_yaw'] = self.yaw
                self.saved_patrol_state['current_segment_traveled_distance'] = self.current_segment_traveled_distance
                self.saved_patrol_state['target_segment_length'] = self.target_segment_length

                remaining_dist_for_log = self.target_segment_length - self.current_segment_traveled_distance
                self.get_logger().info(f"{RED}📦 수동 정지로 정지 전 순찰 상태 저장 완료: "
                                        f"코너 {self.current_patrol_idx+1}, "
                                        f"서브상태 '{self.patrol_motion_state}', "
                                        f"남은거리: {remaining_dist_for_log:.2f}m (원래 길이: {self.patrol_forward_length:.2f}m){RESET}")

        elif self.main_state == 'STOPPED':
            if not self.manual_stop_requested: # 정지 상태인데 수동 정지 요청이 해제된 경우 (재개)
                self.main_state = 'PATROL'
                self.stopped_reason = ""
                self.log_once(GREEN, f"▶️ 정지 해제! (수동 재개). 순찰 재개.")

                # 저장된 상태를 복원하여 순찰 재개
                self.current_patrol_idx = self.saved_patrol_state['current_patrol_idx']
                self.patrol_motion_state = self.saved_patrol_state['patrol_motion_state']

                if self.saved_patrol_state['stop_pose']:
                    # 정지된 지점을 새로운 '시작점'으로 간주하여 남은 거리 계산
                    self.segment_start_pose = copy.deepcopy(self.saved_patrol_state['stop_pose'])
                    self.segment_start_yaw = self.saved_patrol_state['stop_yaw']
                    self.current_segment_traveled_distance = 0.0 # 재개 시점부터 다시 0으로 시작

                    remaining_dist = self.saved_patrol_state['target_segment_length'] - self.saved_patrol_state['current_segment_traveled_distance']
                    self.target_segment_length = max(0.0, remaining_dist) # 음수 방지

                    self.log_once(BLUE, f"🔄 순찰 상태 복원 및 재개 지점 설정 완료. "
                                        f"원래 목표 {self.saved_patrol_state['target_segment_length']:.2f}m 중 "
                                        f"{self.saved_patrol_state['current_segment_traveled_distance']:.2f}m 이동, "
                                        f"이제부터 {self.target_segment_length:.2f}m 더 이동합니다.{RESET}")
                else: # 만약 stop_pose가 설정되지 않은 경우
                    self.segment_start_pose = copy.deepcopy(self.saved_patrol_state['segment_start_pose'])
                    self.segment_start_yaw = self.saved_patrol_state['segment_start_yaw']
                    self.current_segment_traveled_distance = self.saved_patrol_state['current_segment_traveled_distance']
                    self.target_segment_length = self.saved_patrol_state['target_segment_length']


                log_remaining_dist = self.target_segment_length - self.current_segment_traveled_distance
                self.log_once(BLUE, f"🔄 순찰 상태 복원 완료. (코너 {self.current_patrol_idx+1}, 서브상태 '{self.patrol_motion_state}', 최종 남은 이동 거리: {log_remaining_dist:.2f}m){RESET}")

            else: # 여전히 수동 정지 요청 상태
                reason_log = ""
                if self.stopped_reason == "manual":
                    reason_log = "수동 정지 활성화됨"
                else:
                    reason_log = "알 수 없는 이유로 정지"

                current_stop_status_msg = f"⏳ 로봇 정지 중... ({reason_log})"
                self.log_once(YELLOW, current_stop_status_msg)

        # --- 주요 상태 머신 로직 ---
        if self.main_state == 'INITIALIZING':
            pass # Odom 콜백에서 초기화가 완료되면 PATROL 상태로 전환됨

        elif self.main_state == 'PATROL':
            target_yaw_at_corner = self.patrol_absolute_target_yaws[self.current_patrol_idx]

            if self.patrol_motion_state == 'TURN':
                self.log_once(MAGENTA, f"🔄 코너 회전 중... (현재 {self.current_patrol_idx+1}번 코너. 목표 각도: {math.degrees(target_yaw_at_corner):.2f}도)")

                yaw_error = self.normalize_angle(target_yaw_at_corner - self.yaw)

                if abs(yaw_error) > self.patrol_yaw_tolerance:
                    target_angular_z = self.patrol_turn_speed * (yaw_error / abs(yaw_error))
                    target_linear_x = 0.0
                else:
                    self.patrol_motion_state = 'FORWARD'
                    if self.pose: # 회전 완료 후 새로운 직진 구간 시작점을 현재 위치로 설정
                        self.segment_start_pose = copy.deepcopy(self.pose)
                        self.segment_start_yaw = self.yaw
                    self.current_segment_traveled_distance = 0.0
                    self.target_segment_length = self.patrol_forward_length
                    self.log_once(GREEN, f"▶️ 직진 시작. ({self.current_patrol_idx+1}번 코너 방향)")

            elif self.patrol_motion_state == 'FORWARD':
                distance_remaining_in_segment = self.target_segment_length - self.current_segment_traveled_distance

                if distance_remaining_in_segment <= 0.01: # 한 변의 이동이 거의 완료되면
                    target_linear_x = 0.0
                    target_angular_z = 0.0

                    self.patrol_motion_state = 'TURN' # 다음은 회전 단계로
                    self.current_patrol_idx = (self.current_patrol_idx + 1) % len(self.patrol_absolute_target_yaws) # 다음 코너 인덱스
                    self.log_once(GREEN, f"🏁 한 변 이동 완료. 다음 회전 준비 (다음 목표: {math.degrees(self.patrol_absolute_target_yaws[self.current_patrol_idx]):.2f}도, 다음 코너: {self.current_patrol_idx+1}번)")
                    self.current_segment_traveled_distance = 0.0 # 새로운 변 시작이므로 이동 거리 초기화
                    self.target_segment_length = self.patrol_forward_length # 목표 길이도 초기화

                else: # 직진 중
                    target_linear_x = self.patrol_forward_speed

                    # 직진 중 경로 보정 (측면 편차 보정)
                    # 현재 위치에서 세그먼트 시작점을 기준으로 이상적인 목표 위치를 계산
                    # 이 로직은 로봇이 직선 경로에서 벗어나는 것을 방지합니다.
                    ideal_segment_target_x = self.segment_start_pose.x + self.target_segment_length * math.cos(self.segment_start_yaw)
                    ideal_segment_target_y = self.segment_start_pose.y + self.target_segment_length * math.sin(self.segment_start_yaw)

                    dx_to_ideal_target = ideal_segment_target_x - self.pose.x
                    dy_to_ideal_target = ideal_segment_target_y - self.pose.y

                    target_angle_for_segment = math.atan2(dy_to_ideal_target, dx_to_ideal_target)
                    yaw_error_for_segment = self.normalize_angle(target_angle_for_segment - self.yaw)

                    if abs(yaw_error_for_segment) < self.patrol_yaw_tolerance:
                        target_angular_z = 0.0
                    else:
                        target_angular_z = self.patrol_forward_correction_gain * yaw_error_for_segment

                    current_forward_status_msg = (
                        f"🏃 직진 중... (현재 {self.current_patrol_idx+1}번 코너 방향. "
                        f"이동 거리: {self.current_segment_traveled_distance:.2f}/{self.target_segment_length:.2f}m, "
                        f"남은 거리: {self.target_segment_length - self.current_segment_traveled_distance:.2f}m, 경로 보정 각도: {math.degrees(yaw_error_for_segment):.2f}도)"
                    )
                    self.log_once(CYAN, current_forward_status_msg)

        elif self.main_state == 'STOPPED':
            target_linear_x = 0.0
            target_angular_z = 0.0
            if prev_main_state != 'STOPPED': # 정지 상태로 방금 진입했다면 속도 초기화
                self.current_linear_x = 0.0
                self.current_angular_z = 0.0

        # --- 속도 스무딩 로직 ---
        # 갑작스러운 속도 변화를 막아 로봇의 움직임을 부드럽게 합니다.
        twist = Twist()
        delta_linear_x = target_linear_x - self.current_linear_x
        max_delta_linear = self.linear_accel_limit * self.control_loop_dt
        if abs(delta_linear_x) > max_delta_linear:
            twist.linear.x = self.current_linear_x + (max_delta_linear if delta_linear_x > 0 else -max_delta_linear)
        else:
            twist.linear.x = target_linear_x

        delta_angular_z = target_angular_z - self.current_angular_z
        max_delta_angular = self.angular_accel_limit * self.control_loop_dt
        if abs(delta_angular_z) > max_delta_angular:
            twist.angular.z = self.current_angular_z + (max_delta_angular if delta_angular_z > 0 else -max_delta_angular)
        else:
            twist.angular.z = target_angular_z

        self.current_linear_x = twist.linear.x
        self.current_angular_z = twist.angular.z

        self.publisher_.publish(twist)

    def normalize_angle(self, angle):
        """각도를 -π ~ π 범위로 정규화합니다."""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle

def main(args=None):
    rclpy.init(args=args)
    node = SquarePatrolWithoutObstacleAvoidance()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('🛑 종료됨 (Ctrl+C)')
    except SystemExit as e:
        node.get_logger().error(f'🚨 노드 비정상 종료: {e}')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()