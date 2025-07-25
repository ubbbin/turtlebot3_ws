import rclpy
from rclpy.node import Node
# from sensor_msgs.msg import Image
from sensor_msgs.msg import CompressedImage
import cv2
from cv_bridge import CvBridge
import numpy as np
import os
import datetime

# STOP 토픽 메시지 타입을 위한 임포트 추가
from std_msgs.msg import Empty # Empty 메시지 타입은 데이터 없이 시그널만 보낼 때 유용합니다.

class TurtlebotCameraCapture(Node):
    def __init__(self):
        super().__init__('turtlebot_camera_capture')

        # self.camera_topic = '/camera/image_raw'               # 비압축 이미지 토픽
        self.camera_topic = '/camera/image_raw/compressed'      # 압축 이미지 토픽

        self.sub_image = self.create_subscription(
            #Image,
            CompressedImage,
            self.camera_topic,
            self.image_callback,
            10
        )
        self.get_logger().info(f'"{self.camera_topic}" 토픽 구독 시작.')

        self.cv_bridge = CvBridge()
        self.current_frame = None # 현재 프레임을 저장할 변수

        self.base_output_dir = os.path.join(os.path.expanduser('~'), "turtlebot_captured_images") #이미지 저장 경로
        if not os.path.exists(self.base_output_dir):
            try:
                os.makedirs(self.base_output_dir)
                self.get_logger().info(f"'{self.base_output_dir}' 디렉토리를 생성했습니다.")
            except OSError as e:
                self.get_logger().error(f"디렉토리 생성 오류: {self.base_output_dir} - {e}. 권한을 확인하십시오!")
                # 디렉토리 생성 실패 시, 프로그램이 계속 실행될 수 있도록 (이미지 저장은 안 됨)
                self.base_output_dir = None # 저장 디렉토리가 없음을 표시

        # --- STOP 토픽 구독자 추가 ---
        self.stop_subscription = self.create_subscription(
            Empty,            # Empty 메시지 타입 구독
            '/stop_signal',   # 구독할 STOP 토픽 이름 (변경 가능)
            self.stop_callback, # STOP 메시지 수신 시 호출될 콜백 함수
            10
        )
        self.get_logger().info(f"'/stop_signal' 토픽 구독 시작. 이 토픽이 발행되면 이미지가 저장됩니다.")


        self.get_logger().info("터틀봇 카메라 캡처 노드가 시작되었습니다.")
        self.get_logger().info("카메라 캡처를 트리거하려면 '/stop_signal' 토픽을 발행하세요 (예: ros2 topic pub /stop_signal std_msgs/msg/Empty '{}').")
        self.get_logger().info("ROS 2 터미널에서 Ctrl+C를 눌러 노드를 종료하십시오.")

    def image_callback(self, msg):
        """
        카메라 이미지 메시지가 수신될 때 호출되는 콜백 함수.
        """
        try:
            # 메시지 타입에 따라 변환
            # if isinstance(msg, Image):
            #   self.current_frame = self.cv_bridge.imgmsg_to_cv2(msg, 'bgr8')
            if isinstance(msg, CompressedImage):
                 np_arr = np.frombuffer(msg.data, np.uint8)
                 self.current_frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            # (선택 사항) 이미지 처리 및 화면 표시. ROS 2 환경에서는 rviz2 등으로 확인하는 것이 일반적입니다.
            # cv2.imshow("Turtlebot Camera Feed", self.current_frame)
            # cv2.waitKey(1) # GUI 창을 표시한다면 필요

        except Exception as e:
            self.get_logger().error(f"이미지 변환 오류: {e}")
            self.current_frame = None # 오류 발생 시 현재 프레임 초기화

    # --- STOP 토픽을 위한 새로운 콜백 함수 추가 ---
    def stop_callback(self, msg):
        """
        'STOP' 토픽 메시지가 수신될 때 호출되는 콜백 함수.
        이 함수가 호출되면 현재 카메라 프레임을 저장합니다.
        """
        self.get_logger().info("STOP 신호 수신! 현재 카메라 프레임을 저장합니다.")
        self.save_current_frame() # STOP 신호가 오면 이미지 저장 메서드 호출


    def save_current_frame(self):
        """
        현재 프레임을 파일로 저장하는 메서드.
        날짜별 폴더를 생성하고 그 안에 이미지를 저장합니다.
        """
        # 기본 저장 디렉토리가 유효하고 현재 프레임이 있는 경우에만 저장 시도
        if self.current_frame is not None and self.base_output_dir is not None:
            # 현재 날짜를 'YY_MM_DD' 형식으로 가져옵니다.
            today_date_str = datetime.datetime.now().strftime("%y-%m-%d")

            # 날짜별 하위 디렉토리 경로 생성
            date_specific_dir = os.path.join(self.base_output_dir, today_date_str)

            # 날짜별 디렉토리가 없으면 생성
            if not os.path.exists(date_specific_dir):
                try:
                    os.makedirs(date_specific_dir)
                    self.get_logger().info(f"날짜별 디렉토리 '{date_specific_dir}'를 생성했습니다.")
                except OSError as e:
                    self.get_logger().error(f"날짜별 디렉토리 생성 오류: {date_specific_dir} - {e}. 권한을 확인하십시오!")
                    return # 디렉토리 생성 실패 시 저장 중단

            # 파일 이름에 시간 정보 포함 (콜론 제거)
            timestamp = datetime.datetime.now().strftime("%H-%M-%S") # 시간만으로 파일명 생성
            filename = os.path.join(date_specific_dir, f"capture_{timestamp}.jpg") # 날짜 폴더 안에 저장

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
    node = TurtlebotCameraCapture()

    # --- 메인 루프 변경: rclpy.spin()을 사용하여 콜백 기반으로 작동 ---
    # rclpy.spin()은 노드의 모든 콜백 함수가 실행되도록 무한정 블로킹합니다.
    # 이제 이미지 저장은 'STOP' 토픽이 발행될 때 트리거됩니다.
    print("\n--- 터틀봇 카메라 캡처 노드 ---")
    print("노드가 실행 중입니다. '/stop_signal' 토픽이 발행되면 이미지가 저장됩니다.")
    print("ROS 2 터미널에서 'ros2 topic pub /stop_signal std_msgs/msg/Empty '{}'' 명령어를 실행하여 캡처를 트리거하세요.")
    print("노드를 종료하려면 Ctrl+C를 누르십시오.")

    try:
        rclpy.spin(node) # 노드를 스핀하여 콜백을 계속 처리합니다.

    except KeyboardInterrupt:
        node.get_logger().info('노드 종료 요청 (Ctrl+C).')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()