 #최종 파일,사각형으로 돌다가 물체 인식시 접근, 캡쳐 후 물체 인식 풀고 정지, 재개 코드 사용시 복귀 후 본래 사각형 루트 도는 코드
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Twist, Point
from nav_msgs.msg import Odometry
from tf_transformations import euler_from_quaternion
from cv_bridge import CvBridge
import cv2
import numpy as np
import os
import datetime
from std_msgs.msg import Empty
import math
import copy

# 색상 코드 정의
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
CYAN = '\033[96m'
MAGENTA = '\033[95m'
RESET = '\033[0m'

class TurtlebotObjectAligner(Node):
    # --- 로봇 상태 정의 ---
    STATE_INITIALIZING = 0        # 초기화 중 (Odom 대기)
    STATE_PATROLLING = 1          # 순찰 중 (Square Patrol 로직)
    STATE_ALIGNING = 2            # 객체 정렬 및 접근 중
    STATE_STOPPED_WAITING = 3     # 객체 정렬 완료 후 정지 대기 중 (STOP 신호 기다림)
    STATE_RETURNING = 4           # 복귀 중

    def __init__(self):
        super().__init__('turtlebot_object_aligner_node')
        self.get_logger().info("Turtlebot Object Aligner Node has been started.")

        self.bridge = CvBridge()

        # --- ROS 2 Subscribers & Publishers ---
        self.camera_topic = 'camera/image_raw/compressed'
        self.create_subscription(CompressedImage, self.camera_topic, self.image_callback, 10)
        self.get_logger().info(f'"{self.camera_topic}" 토픽 구독 시작.')

        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.get_logger().info("'/odom' 토픽 구독 시작.")

        # /stop_signal은 Empty 메시지로 받아서 복귀 트리거
        self.create_subscription(Empty, '/stop_signal', self.stop_signal_callback, 10)
        self.get_logger().info(f"'/stop_signal' 토픽 구독 시작. STOP 수신 시 복귀 동작 시작됩니다.")

        self.publisher_cmd_vel = self.create_publisher(Twist, 'cmd_vel', 10)
        self.get_logger().info("'cmd_vel' 토픽 퍼블리셔 생성.")

        # Control Loop Timer (전체 제어 로직을 0.05초마다 실행 - 더 부드러운 제어)
        self.control_loop_dt = 0.05
        self.timer = self.create_timer(self.control_loop_dt, self.control_loop)

        # --- 로봇 상태 변수 ---
        self.current_state = self.STATE_INITIALIZING # 초기 상태는 초기화 대기

        # --- Odom 및 위치/방향 변수 ---
        self.pose = None # Point 메시지 (x, y, z)
        self.yaw = 0.0 # 현재 Yaw (라디안)
        self._odom_initialized = False # Odom 초기화 플래그
        self._last_odom_time = self.get_clock().now() # Odom 타임아웃 감지용
        self.sensor_timeout_seconds = 10.0 # Odom 타임아웃 시간

        # --- Square Patrol 로직 변수 (SquarePatrolWithoutObstacleAvoidance에서 가져옴) ---
        self.patrol_motion_state = 'IDLE' # 'FORWARD', 'TURN'
        self.current_patrol_idx = 0 # 현재 사각형 변의 인덱스 (0~3)

        self.patrol_absolute_target_yaws = [0.0, 0.0, 0.0, 0.0]
        self._initial_yaw_offset = None # 초기 Yaw 오프셋

        self.patrol_forward_speed = 0.3
        self.patrol_turn_speed = 0.4
        self.patrol_forward_length = 0.5 # 한 변의 길이 (미터)
        self.patrol_yaw_tolerance = 0.01 # 각도 허용 오차 (라디안)
        self.patrol_forward_correction_gain = 3.0 # 직진 경로 보정 게인

        # 직진 거리 제어 변수 (Square Patrol에서 가져옴)
        self.segment_start_pose = None
        self.segment_start_yaw = 0.0
        self.current_segment_traveled_distance = 0.0
        self.target_segment_length = self.patrol_forward_length # 직진 목표 길이

        # --- 객체 정렬 관련 변수 ---
        self.align_kp_angular = 0.005 # 객체 정렬 Kp (각도)
        self.align_kp_linear = 0.00005 # 객체 정렬 Kp (선형)
        self.target_x = 0 # 이미지 중앙 x 좌표
        self.target_object_area = 20000 # 객체에 얼마나 가까이 갈지 결정하는 면적 (픽셀)
        self.image_width = 0
        self.image_height = 0
        self.current_frame = None # 현재 카메라 프레임

        self.angular_alignment_threshold = 20 # 픽셀
        self.linear_alignment_threshold = 10000 # 픽셀 면적 오차
        self.is_angular_aligned = False
        self.is_linear_aligned = False
        self.last_detected_object_roi = None # 마지막으로 감지된 물체의 ROI (x, y, w, h)

        # --- 복귀 메커니즘 변수 ---
        self.total_linear_offset = 0.0 # '객체 정렬 시작점에서 최종 정지 지점까지 이동한 거리'
        self.return_start_x = 0.0 # 복귀 시작 지점 (정지했던 곳)
        self.return_start_y = 0.0
        self.return_linear_speed = 0.1 # 복귀 선형 속도 (후진 속도)

        # 복귀 타이머 (선형 복귀에만 사용)
        self.linear_return_timer = None

        # 객체 처리가 한 번 완료되었는지 여부
        self._object_handled = False
        # 객체 처리 완료 후 "더 이상 인식 안함" 메시지 출력 여부
        self._object_handled_msg_logged = False

        # 선형 복귀 시작 메시지가 출력되었는지 여부
        self._linear_return_started_log = False
        # 복귀 중 현재 후진한 거리
        self.current_return_traveled_distance = 0.0
        # 로봇이 정지했던 위치의 yaw
        self.stop_yaw_at_object = 0.0


        # --- 로깅 및 상태 관리 플래그 (세밀한 로그 제어용) ---
        self.last_status_msg = ""
        self.warned_no_object = False
        self.was_angular_aligning = False
        self.was_linear_approaching = False
        self.was_fully_aligned = False

        # --- 상태 저장 변수 (정지/재개 시 순찰 상태 복원용) ---
        self.saved_patrol_state = {
            'patrol_motion_state': 'IDLE',
            'current_patrol_idx': 0,
            'segment_start_pose': None,
            'segment_start_yaw': 0.0,
            'current_segment_traveled_distance': 0.0,
            'target_segment_length': 0.0,
            'stop_pose': None, # 정지된 정확한 위치
            'stop_yaw': 0.0 # 정지된 정확한 Yaw
        }

        # For smooth acceleration/deceleration
        self.current_linear_x = 0.0
        self.current_angular_z = 0.0
        self.linear_accel_limit = 0.5
        self.angular_accel_limit = 1.0

        # --- 이미지 저장 경로 ---
        self.base_output_dir = os.path.join(os.path.expanduser('~'), "turtlebot_captured_images")
        if not os.path.exists(self.base_output_dir):
            try:
                os.makedirs(self.base_output_dir)
                self.get_logger().info(f"기본 저장 디렉토리 '{self.base_output_dir}'를 생성했습니다.")
            except OSError as e:
                self.get_logger().error(f"디렉토리 생성 오류: {self.base_output_dir} - {e}. 권한을 확인하십시오!")
                self.base_output_dir = None

        # OpenCV 창 이름을 미리 정의
        self.window_name = "Turtlebot3 Camera Feed with Object Alignment"
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)

        self.get_logger().info("터틀봇 객체 정렬 노드가 시작되었습니다.")
        self.get_logger().info("초기 상태: Odom 데이터 대기 중...")
        self.get_logger().info("물체 감지 후 정지 상태에서 복귀를 시작하려면 '/stop_signal' 토픽을 발행하세요 (예: ros2 topic pub -1 /stop_signal std_msgs/msg/Empty '{}' --once).")
        self.get_logger().info("ROS 2 터미널에서 Ctrl+C를 눌러 노드를 종료하십시오.")

    def log_once(self, color, msg):
        """이전과 동일한 메시지는 다시 로깅하지 않아 메시지 스팸을 방지합니다."""
        if self.last_status_msg != msg:
            self.get_logger().info(f"{color}{msg}{RESET}")
            self.last_status_msg = msg

    # --- Odom 콜백 (Square Patrol에서 가져옴 + 오프셋 누적 로직 포함) ---
    def odom_callback(self, msg):
        self._last_odom_time = self.get_clock().now()

        self.pose = msg.pose.pose.position
        orientation_q = msg.pose.pose.orientation

        _, _, current_absolute_yaw = euler_from_quaternion([
            orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w
        ])
        self.yaw = current_absolute_yaw

        # PATROL 상태에서 직진 중일 때만 이동 거리 업데이트
        if self.current_state == self.STATE_PATROLLING and self.patrol_motion_state == 'FORWARD' and self.segment_start_pose:
            self.current_segment_traveled_distance = math.sqrt(
                (self.pose.x - self.segment_start_pose.x)**2 +
                (self.pose.y - self.segment_start_pose.y)**2
            )

        # ALIGNING 상태일 때만 오프셋 누적 (복귀 시 사용)
        if self.current_state == self.STATE_ALIGNING and self.segment_start_pose:
            # total_linear_offset 업데이트: 정렬 시작점부터 현재 로봇 위치까지의 거리
            dx_from_start = self.pose.x - self.segment_start_pose.x
            dy_from_start = self.pose.y - self.segment_start_pose.y
            self.total_linear_offset = math.sqrt(dx_from_start**2 + dy_from_start**2)

        # RETURNING 상태일 때 후진한 거리 업데이트
        if self.current_state == self.STATE_RETURNING and self.return_start_x is not None:
            # 후진 시작점(정지했던 곳)으로부터 현재까지 이동한 거리
            dx_from_return_start = self.pose.x - self.return_start_x
            dy_from_return_start = self.pose.y - self.return_start_y
            self.current_return_traveled_distance = math.sqrt(dx_from_return_start**2 + dy_from_return_start**2)

        if not self._odom_initialized:
            self._initial_yaw_offset = current_absolute_yaw
            # 초기 오프셋을 기준으로 각 사각형 모서리의 절대 각도 계산
            # Square_bot.py의 로직을 그대로 가져옴
            self.patrol_absolute_target_yaws = [
                self.normalize_angle(self._initial_yaw_offset + math.radians(0)), # 첫 번째 직진 방향 (초기 방향)
                self.normalize_angle(self._initial_yaw_offset + math.radians(-90)), # 오른쪽으로 90도 회전
                self.normalize_angle(self._initial_yaw_offset + math.radians(-180)), # 또 오른쪽으로 90도 회전 (총 180도)
                self.normalize_angle(self._initial_yaw_offset + math.radians(90)) # 또 오른쪽으로 90도 회전 (총 270도)
            ]
            self.log_once(GREEN, f"🟢 Odom 초기화 완료. 초기 방향: {math.degrees(self._initial_yaw_offset):.2f}도.")
            self.log_once(GREEN, f"🟢 순찰 목표 방향 설정 완료: {[math.degrees(y) for y in self.patrol_absolute_target_yaws]}도")
            self._odom_initialized = True

            self.current_state = self.STATE_PATROLLING
            self.patrol_motion_state = 'FORWARD' # 처음에는 직진부터 시작
            self.current_patrol_idx = 0 # 첫 번째 변을 순찰

            if self.pose:
                self.segment_start_pose = copy.deepcopy(self.pose)
                self.segment_start_yaw = self.yaw
            self.current_segment_traveled_distance = 0.0
            self.target_segment_length = self.patrol_forward_length
            self.log_once(GREEN, "🚶 초기 회전 없이 바로 직진 순찰 시작. (1번 코너 방향)")

    # --- Image Callback (상태 기반 처리) ---
    def image_callback(self, msg):
        try:
            cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, 'bgr8')
            self.current_frame = cv_image

            if self.image_width == 0:
                self.image_height, self.image_width, _ = cv_image.shape
                self.target_x = self.image_width // 2
                self.get_logger().info(f"이미지 해상도: {self.image_width}x{self.image_height}, 중앙 X: {self.target_x}")

            processed_image = cv_image.copy() # 원본 이미지를 복사하여 작업

            # 물체 감지 (어떤 상태에서든 감지는 계속)
            object_center_x, object_area, object_roi = self.detect_and_get_object_info(processed_image)

            # _object_handled 플래그가 False일 때만 객체 감지 및 정렬 모드 진입
            if self.current_state == self.STATE_PATROLLING and not self._object_handled:
                if object_center_x is not None:
                    # 순찰 중 객체 발견 시 객체 추적 모드로 전환
                    self.get_logger().info(f"{YELLOW}객체 발견! 순찰 중단, 객체 정렬 모드 진입.{RESET}")

                    # 현재 순찰 상태 저장 (복귀 후 순찰 재개 시 사용)
                    self.saved_patrol_state['patrol_motion_state'] = self.patrol_motion_state
                    self.saved_patrol_state['current_patrol_idx'] = self.current_patrol_idx
                    self.saved_patrol_state['segment_start_pose'] = copy.deepcopy(self.segment_start_pose)
                    self.saved_patrol_state['segment_start_yaw'] = self.segment_start_yaw
                    self.saved_patrol_state['current_segment_traveled_distance'] = self.current_segment_traveled_distance
                    self.saved_patrol_state['target_segment_length'] = self.target_segment_length

                    self.current_state = self.STATE_ALIGNING
                    self.stop_robot() # 정렬 시작 전 잠시 멈춤

                    # 정렬을 위한 새로운 시작점 (현재 위치) 저장
                    # **중요**: 이 위치가 후진 복귀 시의 '원래 장애물 발견 위치'가 됩니다.
                    if self.pose:
                        self.segment_start_pose = copy.deepcopy(self.pose) # 정렬 시작점
                        self.segment_start_yaw = self.yaw # 정렬 시작 시의 Yaw

                    # 객체 처리 완료 메시지 플래그 초기화
                    self._object_handled_msg_logged = False

            # _object_handled이 True인 경우 순찰 중에는 객체 감지를 무시
            elif self.current_state == self.STATE_PATROLLING and self._object_handled:
                if not self._object_handled_msg_logged:
                    self.log_once(CYAN, "🔄 순찰 재개. 객체는 더 이상 인식하지 않습니다.")
                    self._object_handled_msg_logged = True
                pass # 객체를 무시하고 순찰 계속

            if self.current_state == self.STATE_ALIGNING:
                # 객체 추적 및 정렬 로직 실행
                self.control_robot_align(object_center_x, object_area, object_roi)
                # 객체가 화면에 보일 때는 경계 상자 그림
                if object_roi is not None:
                    x, y, w, h = object_roi
                    cv2.rectangle(processed_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.putText(processed_image, "Black Object", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    cv2.circle(processed_image, (object_center_x, y + h // 2), 5, (0, 255, 255), -1)

            elif self.current_state == self.STATE_STOPPED_WAITING:
                # 정지 대기 중: 마지막으로 감지된 ROI를 검정색으로 덮기 (장애물 인식 끄기 효과)
                if self.last_detected_object_roi is not None:
                    x, y, w, h = self.last_detected_object_roi
                    x1 = max(0, x)
                    y1 = max(0, y)
                    x2 = min(self.image_width, x + w)
                    y2 = min(self.image_height, y + h)
                    if x1 < x2 and y1 < y2:
                        cv2.rectangle(processed_image, (x1, y1), (x2, y2), (0, 0, 0), -1) # 검정색으로 채우기
                self.stop_robot() # 안전을 위해 계속 정지 명령 발행

            elif self.current_state == self.STATE_RETURNING:
                # 복귀 중: 복귀 타이머가 제어를 담당하므로 여기서는 화면만 보여줌
                pass # 복귀 중에는 물체 감지 화면 그대로 표시

            cv2.imshow(self.window_name, processed_image)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f"Failed to process image: {e}. Stopping robot.")
            self.current_frame = None
            self.stop_robot()
            if self.image_width > 0 and self.image_height > 0:
                black_image = np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8)
                cv2.imshow(self.window_name, black_image)
                cv2.waitKey(1)

    # --- 주요 제어 루프 (모든 상태 관리) ---
    def control_loop(self):
        current_time = self.get_clock().now()
        target_linear_x = 0.0
        target_angular_z = 0.0

        time_since_last_odom = (current_time - self._last_odom_time).nanoseconds / 1e9

        if time_since_last_odom > self.sensor_timeout_seconds:
            self.get_logger().error(f"{RED}❌ 치명적 오류: {self.sensor_timeout_seconds}초 이상 Odom 데이터 미수신! 노드를 종료합니다.{RESET}")
            # ROS 2 노드 종료를 위해 rclpy.shutdown() 호출
            rclpy.shutdown()
            raise SystemExit("Odom data timeout, exiting node.")

        if not self._odom_initialized or self.pose is None:
            if (current_time - self._last_odom_time).nanoseconds / 1e9 >= 5.0:
                self.get_logger().warn(f"{YELLOW}⚠️ 필수 데이터(Odom/Pose) 수신 대기 중... Odom 초기화: {self._odom_initialized}, Pose 유효: {self.pose is not None}{RESET}")
            self.stop_robot() # 데이터를 받기 전까지는 로봇을 정지
            return

        # --- 메인 상태 머신 ---
        if self.current_state == self.STATE_INITIALIZING:
            # Odom 콜백에서 초기화가 완료되면 STATE_PATROLLING으로 전환됨
            self.log_once(YELLOW, "⏳ Odom 초기화 및 순찰 준비 중...")
            target_linear_x = 0.0
            target_angular_z = 0.0

        elif self.current_state == self.STATE_PATROLLING:
            # Square Patrol 로직 (SquarePatrolWithoutObstacleAvoidance에서 가져옴)
            target_yaw_at_corner = self.patrol_absolute_target_yaws[self.current_patrol_idx]

            if self.patrol_motion_state == 'TURN':
                self.log_once(MAGENTA, f"🔄 코너 회전 중... (현재 {self.current_patrol_idx+1}번 코너. 목표 각도: {math.degrees(target_yaw_at_corner):.2f}도)")

                yaw_error = self.normalize_angle(target_yaw_at_corner - self.yaw)

                if abs(yaw_error) > self.patrol_yaw_tolerance:
                    target_angular_z = self.patrol_turn_speed * (yaw_error / abs(yaw_error))
                    target_linear_x = 0.0
                else:
                    self.patrol_motion_state = 'FORWARD'
                    # 회전 완료 후 새로운 직진 구간 시작점을 현재 위치로 설정
                    if self.pose:
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
                    if self.segment_start_pose:
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
                    else:
                        target_angular_z = 0.0 # segment_start_pose가 없으면 보정 안함 (초기 상태)


                    current_forward_status_msg = (
                        f"🏃 직진 중... (현재 {self.current_patrol_idx+1}번 코너 방향. "
                        f"이동 거리: {self.current_segment_traveled_distance:.2f}/{self.target_segment_length:.2f}m, "
                        f"남은 거리: {self.target_segment_length - self.current_segment_traveled_distance:.2f}m, "
                        f"경로 보정 각도: {math.degrees(target_angular_z):.2f}도)"
                    )
                    self.log_once(CYAN, current_forward_status_msg)

        elif self.current_state == self.STATE_ALIGNING:
            # 객체 정렬 로직 (image_callback에서 제어 메시지 발행)
            self.log_once(BLUE, "✨ 객체 정렬 및 접근 중...")
            target_linear_x = self.current_linear_x # control_robot_align에서 설정된 값 유지
            target_angular_z = self.current_angular_z # control_robot_align에서 설정된 값 유지

        elif self.current_state == self.STATE_STOPPED_WAITING:
            self.log_once(YELLOW, "⏳ 객체 정렬 완료. STOP 신호 대기 중...")
            target_linear_x = 0.0
            target_angular_z = 0.0

        elif self.current_state == self.STATE_RETURNING:
            self.log_once(MAGENTA, "↩️ 복귀 중...")
            # 복귀 로직은 linear_return_timer_callback에서 처리
            target_linear_x = self.current_linear_x # 복귀 타이머에서 설정된 값 유지
            target_angular_z = self.current_angular_z # 복귀 타이머에서 설정된 값 유지

        # --- 속도 스무딩 로직 ---
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

        # STOPPED_WAITING 또는 RETURNING 상태에서는 타이머가 직접 publish하므로, control_loop에서는 발행하지 않음
        # 단, current_state가 ALIGNING, PATROLLING, INITIALIZING 일 때만 발행
        if self.current_state not in [self.STATE_STOPPED_WAITING, self.STATE_RETURNING]:
            self.publisher_cmd_vel.publish(twist)
        elif self.current_state == self.STATE_STOPPED_WAITING:
            self.stop_robot() # 명시적으로 정지 명령 유지

    # --- 객체 감지 및 정보 반환 ---
    def detect_and_get_object_info(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 255, 50])
        mask = cv2.inRange(hsv, lower_black, upper_black)
        kernel = np.ones((5,5),np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=1)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        largest_area = 0
        target_object_center_x = None
        target_object_area = 0
        target_object_roi = None

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 500: # 최소 면적 이상만 고려
                x, y, w, h = cv2.boundingRect(cnt)
                center_x = x + w // 2
                if area > largest_area:
                    largest_area = area
                    target_object_center_x = center_x
                    target_object_area = area
                    target_object_roi = (x, y, w, h)
        return target_object_center_x, target_object_area, target_object_roi

    # --- 로봇 제어 (객체 정렬 및 접근) ---
    def control_robot_align(self, object_center_x, object_area, object_roi):
        twist_msg = Twist()
        current_was_angular_aligning = False
        current_was_linear_approaching = False
        current_was_fully_aligned = False

        if object_center_x is not None and self.image_width > 0:
            self.warned_no_object = False
            error_x = self.target_x - object_center_x

            # 1단계: 앵귤러 정렬
            if not self.is_angular_aligned:
                twist_msg.angular.z = self.align_kp_angular * error_x
                twist_msg.linear.x = 0.0
                max_angular_vel = 0.5
                twist_msg.angular.z = np.clip(twist_msg.angular.z, -max_angular_vel, max_angular_vel)

                if abs(error_x) < self.angular_alignment_threshold:
                    self.is_angular_aligned = True
                    if self.was_angular_aligning: # 정렬 완료 시에만 로그
                        self.get_logger().info(f"{GREEN}🎯 앵귤러 정렬 완료! 이제 리니어 접근 시작.{RESET}")
                    self.is_linear_aligned = False
                else:
                    if not self.was_angular_aligning or abs(self.current_angular_z - twist_msg.angular.z) > 0.01:
                        self.get_logger().info(f"🔄 앵귤러 정렬 중 - Object X: {object_center_x} -> Angular: {twist_msg.angular.z:.2f}")
                    current_was_angular_aligning = True
            # 2단계: 리니어 접근 (앵귤러 정렬 완료 후)
            else:
                twist_msg.angular.z = 0.0
                area_error = self.target_object_area - object_area
                twist_msg.linear.x = self.align_kp_linear * area_error
                max_linear_vel = 0.1
                twist_msg.linear.x = np.clip(twist_msg.linear.x, -max_linear_vel, max_linear_vel)

                if abs(area_error) < self.linear_alignment_threshold:
                    self.is_linear_aligned = True
                    if self.was_linear_approaching: # 접근 완료 시에만 로그
                        self.get_logger().info(f"{GREEN}📏 리니어 접근 완료!{RESET}")
                else:
                    if not self.was_linear_approaching or abs(self.current_linear_x - twist_msg.linear.x) > 0.005:
                        self.get_logger().info(f"🏃 리니어 접근 중 - Area: {object_area} -> Linear: {twist_msg.linear.x:.2f}")
                    current_was_linear_approaching = True

            # 최종 정렬 완료 상태 확인 및 정지
            if self.is_angular_aligned and self.is_linear_aligned:
                twist_msg.angular.z = 0.0
                twist_msg.linear.x = 0.0
                if not self.was_fully_aligned:
                    self.get_logger().info(f"{BLUE}✅ 정렬 및 접근 완료! 로봇 정지. STOP 신호를 기다립니다.{RESET}")
                    self.stop_robot()
                    # 물체에 가까이 갔을 때 물체 캡처 (record_stop_location 함수에 포함)
                    self.record_stop_location(object_roi) # 정지 위치 및 ROI 정보 기록
                    self.current_state = self.STATE_STOPPED_WAITING # 대기 상태로 전환
                current_was_fully_aligned = True

        else: # 물체 감지 안됨 (ALIGNING 상태인데 물체가 사라졌을 경우)
            if not self.warned_no_object:
                self.get_logger().warn(f"{RED}객체 추적 중 물체 감지 안됨. 순찰 모드로 돌아갑니다.{RESET}")
                self.warned_no_object = True
            self.stop_robot()
            self.reset_alignment_flags() # 정렬 관련 플래그 초기화
            self.current_state = self.STATE_PATROLLING # 순찰 모드로 복귀
            # 순찰 재개 시점의 위치를 새로운 segment_start_pose로 설정
            if self.pose:
                self.segment_start_pose = copy.deepcopy(self.pose)
                self.segment_start_yaw = self.yaw
            self.current_segment_traveled_distance = 0.0
            self.target_segment_length = self.patrol_forward_length
            self.patrol_motion_state = 'FORWARD' # 순찰 재개 시 직진부터

        # cmd_vel 퍼블리싱 (스무딩을 위해 current_linear_x/angular_z에 저장)
        self.current_linear_x = twist_msg.linear.x
        self.current_angular_z = twist_msg.angular.z

        self.was_angular_aligning = current_was_angular_aligning
        self.was_linear_approaching = current_was_linear_approaching
        self.was_fully_aligned = current_was_fully_aligned

    def stop_robot(self):
        stop_twist = Twist()
        stop_twist.linear.x = 0.0
        stop_twist.angular.z = 0.0
        self.publisher_cmd_vel.publish(stop_twist)
        self.current_linear_x = 0.0 # 스무딩 변수도 0으로 초기화
        self.current_angular_z = 0.0

    def record_stop_location(self, object_roi):
        """
        객체 정렬 및 접근 완료 시 로봇의 현재 위치 및 오프셋 정보를 기록하고 ROI를 저장합니다.
        total_linear_offset은 정렬 시작점부터 최종 정지 지점까지 이동한 총 거리가 됩니다.
        **이 함수 내에서 save_current_frame()을 호출하여 이미지를 캡처합니다.**
        """
        if self.pose and self.segment_start_pose:
            # 최종 정지 위치에서 정렬 시작점까지의 거리를 total_linear_offset으로 저장 (이것이 복귀 목표 거리)
            dx = self.pose.x - self.segment_start_pose.x
            dy = self.pose.y - self.segment_start_pose.y
            self.total_linear_offset = math.sqrt(dx**2 + dy**2)
        else:
            self.total_linear_offset = 0.0
            self.get_logger().warn(f"{YELLOW}⚠️ segment_start_pose가 없어 total_linear_offset을 0으로 설정합니다.{RESET}")

        self.get_logger().info(
            f"{BLUE}✅ 객체 정렬 및 접근 완료. 로봇 위치 기록됨:"
            f"\n  X: {self.pose.x:.2f} m"
            f"\n  Y: {self.pose.y:.2f} m"
            f"\n  Yaw: {math.degrees(self.yaw):.2f} degrees"
            f"\n  복귀 목표 거리 (정렬 시작점으로부터): {self.total_linear_offset:.2f} m{RESET}"
        )

        # 복귀를 위해 정지 시점의 위치와 yaw를 저장
        self.return_start_x = self.pose.x
        self.return_start_y = self.pose.y
        self.stop_yaw_at_object = self.yaw # 정지 시 로봇의 Yaw 저장

        self.last_detected_object_roi = object_roi # 감지된 ROI 저장
        self.save_current_frame() # **여기서 객체 정렬 완료 시 현재 프레임 캡처**

    def save_current_frame(self):
        if self.current_frame is not None and self.base_output_dir is not None:
            today_date_str = datetime.datetime.now().strftime("%y-%m-%d")
            date_specific_dir = os.path.join(self.base_output_dir, today_date_str)
            if not os.path.exists(date_specific_dir):
                try:
                    os.makedirs(date_specific_dir)
                    self.get_logger().info(f"날짜별 디렉토리 '{date_specific_dir}'를 생성했습니다.")
                except OSError as e:
                    self.get_logger().error(f"디렉토리 생성 오류: {date_specific_dir} - {e}. 권한을 확인하십시오!")
                    return
            timestamp = datetime.datetime.now().strftime("%H-%M-%S")
            filename = os.path.join(date_specific_dir, f"capture_{timestamp}.jpg")
            try:
                cv2.imwrite(filename, self.current_frame)
                self.get_logger().info(f"{GREEN}📸 물체 정렬 완료 후 이미지 저장됨: {filename}{RESET}")
            except Exception as e:
                self.get_logger().error(f"이미지 저장 오류: {e}. 저장 경로 권한을 확인하십시오!")
        elif self.base_output_dir is None:
            self.get_logger().error("이미지 저장 기본 디렉토리가 유효하지 않습니다. 초기화 오류를 확인하십시오.")
        else:
            self.get_logger().warn("저장할 현재 프레임이 없습니다. 카메라 메시지를 기다리는 중입니다.")

    def reset_alignment_flags(self):
        """정렬 관련 플래그들을 초기 상태로 리셋합니다."""
        self.is_angular_aligned = False
        self.is_linear_aligned = False
        self.was_angular_aligning = False
        self.was_linear_approaching = False
        self.was_fully_aligned = False
        self.warned_no_object = False

    # --- 복귀 관련 콜백 및 함수 ---
    def stop_signal_callback(self, msg):
        """
        STOP 신호 수신 시 로봇을 정지시키고 복귀 프로세스를 시작합니다.
        Empty 메시지를 받으며, STATE_STOPPED_WAITING 상태에서만 유효합니다.
        """
        if self.current_state == self.STATE_RETURNING:
            self.get_logger().warn(f"{YELLOW}⚠️ 이미 복귀 동작 중입니다. STOP 신호 무시.{RESET}")
            return

        if self.current_state != self.STATE_STOPPED_WAITING:
            self.get_logger().warn(f"{YELLOW}⚠️ 현재 상태가 STATE_STOPPED_WAITING이 아닙니다 (현재: {self.current_state}). STOP 신호 무시.{RESET}")
            return

        self.get_logger().info(f"{BLUE}📢 STOP 신호 수신! 복귀 프로세스 시작.{RESET}")
        self.stop_robot() # 즉시 로봇 정지
        self.current_state = self.STATE_RETURNING # 복귀 모드로 전환

        # 기존 타이머가 있으면 파괴
        if self.linear_return_timer:
            self.linear_return_timer.destroy()
            self.linear_return_timer = None

        # 선형 복귀 시작 로깅 플래그 초기화
        self._linear_return_started_log = False
        self.current_return_traveled_distance = 0.0 # 복귀 이동 거리 초기화

        self.start_linear_return() # 바로 선형 복귀 시작

    def start_linear_return(self):
        """
        선형 복귀 프로세스를 시작합니다.
        기록된 `total_linear_offset`만큼 후진합니다.
        """
        if abs(self.total_linear_offset) < 0.01: # 복귀할 거리가 너무 작으면 생략 (1cm 미만)
            self.get_logger().info(f"{GREEN}🟢 선형 복귀 생략 (이동 오프셋이 너무 작음: {self.total_linear_offset:.2f}m).{RESET}")
            self.complete_return_process()
            return

        # 복귀 타이머 생성
        self.linear_return_timer = self.create_timer(0.05, self.linear_return_timer_callback)
        self.get_logger().info(f"{MAGENTA}🏃 후진 복귀 시작. 되돌아갈 거리: {self.total_linear_offset:.2f}m{RESET}")

    def linear_return_timer_callback(self):
        if self.current_state != self.STATE_RETURNING:
            self.get_logger().warn(f"{YELLOW}⚠️ 복귀 중이 아닌데 linear_return_timer_callback이 호출되었습니다. 타이머 중지.{RESET}")
            self.linear_return_timer.destroy()
            self.linear_return_timer = None
            self.stop_robot()
            return

        # 현재 후진해야 할 남은 거리
        distance_to_go_back = self.total_linear_offset - self.current_return_traveled_distance

        twist_msg = Twist()

        if distance_to_go_back <= 0.05: # 5cm 이내면 복귀 완료 (허용 오차)
            twist_msg.linear.x = 0.0
            twist_msg.angular.z = 0.0
            self.get_logger().info(f"{GREEN}✅ 선형 복귀 완료! (총 {self.current_return_traveled_distance:.2f}m 후진){RESET}")
            self.stop_robot()
            self.linear_return_timer.destroy()
            self.linear_return_timer = None
            self._linear_return_started_log = False # 다음 번을 위해 초기화

            self.complete_return_process()

        else:
            # 후진 방향 계산 및 보정 (정지 시 바라보던 방향을 유지하며 후진)
            # 로봇이 정지했을 때의 Yaw (stop_yaw_at_object)를 기준으로 후진합니다.
            angle_error = self.normalize_angle(self.stop_yaw_at_object - self.yaw)

            twist_msg.linear.x = -self.return_linear_speed # 후진을 위해 음수 값 사용

            # 각도 보정: 정지 시의 Yaw와 현재 Yaw의 오차를 줄이도록 회전
            # 후진하면서 로봇의 방향이 틀어지는 것을 보정
            twist_msg.angular.z = self.align_kp_angular * angle_error * 3.0 # 각도 보정 게인 조정 (필요 시 조절)
            max_angular_vel = 0.2 # 후진 중 최대 회전 속도 제한
            twist_msg.angular.z = np.clip(twist_msg.angular.z, -max_angular_vel, max_angular_vel)

            self.publisher_cmd_vel.publish(twist_msg)
            # 이 로그는 너무 자주 출력되므로 주석 처리하거나, 필요시 특정 조건에서만 출력하도록 수정
            # self.get_logger().info(f"🏃 후진 복귀 중. 남은 거리: {distance_to_go_back:.2f} m / 총 이동했던 거리: {self.total_linear_offset:.2f} m")

    def complete_return_process(self):
        """
        복귀 프로세스 완료 후 로봇 상태를 재설정하고 순찰 모드로 복귀합니다.
        """
        self.total_linear_offset = 0.0  # 오프셋 초기화
        self.current_return_traveled_distance = 0.0 # 복귀 이동 거리 초기화
        self.reset_alignment_flags() # 정렬 플래그 초기화

        # 핵심 변경: 객체 처리가 완료되었음을 표시
        self._object_handled = True

        self.current_state = self.STATE_PATROLLING # 순찰 모드로 재개

        # 저장된 순찰 상태를 복원하여 순찰을 중단했던 지점부터 재개
        self.patrol_motion_state = self.saved_patrol_state['patrol_motion_state']
        self.current_patrol_idx = self.saved_patrol_state['current_patrol_idx']
        self.segment_start_pose = copy.deepcopy(self.saved_patrol_state['segment_start_pose'])
        self.segment_start_yaw = self.saved_patrol_state['segment_start_yaw']
        self.current_segment_traveled_distance = self.saved_patrol_state['current_segment_traveled_distance']
        self.target_segment_length = self.saved_patrol_state['target_segment_length']

        self.get_logger().info(f"{GREEN}🔵 복귀 완료! 순찰 모드 재개. (복원된 상태: 코너 {self.current_patrol_idx+1}, 서브상태 '{self.patrol_motion_state}', 남은 이동 거리: {self.target_segment_length - self.current_segment_traveled_distance:.2f}m){RESET}")

        # 이 메시지는 complete_return_process에서만 한 번 출력되도록 플래그를 사용합니다.
        self._object_handled_msg_logged = False # 다시 False로 초기화하여 다음 객체 감지 시 재사용 가능
        self.log_once(CYAN, "🚨 이제 이 노드는 물체를 다시 인식하지 않고 순찰을 계속합니다.")

        self.stop_robot() # 재개 전 잠시 멈춤

    def normalize_angle(self, angle):
        """각도를 -π ~ π 범위로 정규화합니다."""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle

    def destroy_node(self):
        cv2.destroyAllWindows()
        # 모든 타이머 파괴 (노드 종료 시 깔끔하게)
        if hasattr(self, 'timer') and self.timer is not None:
            self.timer.destroy()
        if hasattr(self, 'linear_return_timer') and self.linear_return_timer is not None:
            self.linear_return_timer.destroy()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = TurtlebotObjectAligner()

    print("\n--- 터틀봇 단일 객체 정렬 및 후진 복귀 노드 ---")
    print("  - 이 노드는 **오도메트리 기반의 정사각형 순찰**, 객체 정렬, 그리고 수동 후진 복귀 기능을 모두 포함합니다.")
    print("  - 시작 시 로봇은 지정된 길이로 정사각형 순찰을 시작합니다.")
    print("  - 순찰 중 **검정색 물체**가 감지되면, 로봇은 순찰을 중단하고 물체에 정렬하여 가까이 접근한 후 정지합니다.")
    print("  - 정렬 완료 후 로봇은 정지하며, 카메라 피드에서 감지된 물체 영역은 검정색으로 가려집니다.")
    print("  - **물체 정렬 완료 후 자동으로 현재 카메라 프레임을 한 번 캡처하여 저장합니다.**")
    print("  - 이 상태에서 터미널에서 '/stop_signal' 토픽을 **단 한 번** 발행하면 로봇은 물체를 발견하기 위해 **이동했던 거리만큼 후진**하여 원래 순찰 경로의 지점으로 돌아옵니다.")
    print("    (명령어: **ros2 topic pub -1 /stop_signal std_msgs/msg/Empty '{}' --once**)")
    print("  - 복귀가 시작되면 화면에 가려졌던 물체가 다시 보입니다.")
    print("  - 복귀 완료 후 로봇은 **이전 순찰 상태를 복원하여 중단했던 순찰을 정확히 재개**하며, **더 이상 물체를 인식하지 않습니다.** (이 메시지는 한 번만 출력됩니다.)")
    print("\nROS 2 터미널에서 Ctrl+C를 눌러 노드를 종료하십시오.")

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        node.get_logger().info('🛑 노드 종료 요청 (Ctrl+C).')
    except SystemExit as e:
        node.get_logger().error(f'🚨 노드 비정상 종료: {e}. Odom 데이터 타임아웃 발생 가능성.')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
    #ros2 topic pub -1 /stop_signal std_msgs/msg/Empty '{}' --once 복귀 및 재개
    #최종 파일,사각형으로 돌다가 물체 인식시 접근, 캡쳐 후 물체 인식 풀고 정지, 재개 코드 사용시 복귀 후 본래 사각형 루트 도는 코드