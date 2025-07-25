import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
import cv2
import numpy as np
import os
import datetime
# std_msgs.msg.Empty는 이 코드에서 직접 사용되지는 않지만,
# 이전 코드와의 연계성을 위해 남겨두거나 삭제해도 무방합니다.
from std_msgs.msg import Empty 

class TurtlebotObjectDetector(Node):
    def __init__(self):
        super().__init__('turtlebot_object_detector_node')
        self.get_logger().info("Turtlebot Object Detector Node has been started.")

        self.bridge = CvBridge()

        # 카메라 토픽 구독 설정 (압축 이미지)
        self.camera_topic = 'camera/image_raw/compressed' # 로봇의 실제 카메라 토픽으로 변경 필요!
        self.subscription = self.create_subscription(
            CompressedImage,
            self.camera_topic,
            self.image_callback,
            10)
        self.get_logger().info(f'"{self.camera_topic}" 토픽 구독 시작.')

        self.current_frame = None # 현재 프레임을 저장할 변수

        # 이미지 저장 경로 (이전 코드에서 유지)
        self.base_output_dir = os.path.join(os.path.expanduser('~'), "turtlebot_captured_images")
        if not os.path.exists(self.base_output_dir):
            try:
                os.makedirs(self.base_output_dir)
                self.get_logger().info(f"기본 저장 디렉토리 '{self.base_output_dir}'를 생성했습니다.")
            except OSError as e:
                self.get_logger().error(f"기본 디렉토리 생성 오류: {self.base_output_dir} - {e}. 권한을 확인하십시오!")
                self.base_output_dir = None

        # STOP 토픽 구독자 설정 (이전 코드에서 유지)
        self.stop_subscription = self.create_subscription(
            Empty,
            '/stop_signal',
            self.stop_callback,
            10
        )
        self.get_logger().info(f"'/stop_signal' 토픽 구독 시작. 이 토픽이 발행되면 이미지가 저장됩니다.")


    def image_callback(self, msg):
        try:
            cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, 'bgr8')
            self.current_frame = cv_image # 최신 프레임을 저장

            # --- ROI를 이용한 물체 감지 및 사각형 그리기 로직 ---
            processed_image = self.detect_and_draw_roi(cv_image)

            # 처리된 이미지 화면에 표시
            cv2.imshow("Turtlebot3 Camera Feed with ROI", processed_image)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f"Failed to process image: {e}")
            self.current_frame = None

    def detect_and_draw_roi(self, image):
        """
        이미지에서 특정 색상(파란색)을 감지하고, 해당 영역에 사각형을 그립니다.
        """
        # 1. BGR 이미지를 HSV 색 공간으로 변환 (색상 기반 인식에 유리)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # 2. 파란색 범위 정의 (Hue, Saturation, Value)
        # 이 값들은 환경에 따라 조절해야 할 수 있습니다.
        # 밝은 파란색
        #lower_blue = np.array([100, 100, 100])
        #upper_blue = np.array([130, 255, 255])
        
        # 어두운 파란색이나 다른 색상을 원하면 이 값을 변경하세요.
        # 예시: 초록색 범위
        #lower_green = np.array([40, 40, 40])
        #upper_green = np.array([80, 255, 255])
        
        lower_red = np.array([0, 100, 100])
        upper_red = np.array([10, 255, 255])
        
        
        # 3. 마스크 생성: 정의된 색상 범위에 해당하는 픽셀만 흰색(255), 나머지는 검은색(0)
        mask = cv2.inRange(hsv, lower_red, upper_red)

        # 4. 노이즈 제거 (선택 사항): 모폴로지 연산으로 작은 노이즈 제거 및 영역 채우기
        kernel = np.ones((5,5),np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1) # 침식 (작은 노이즈 제거)
        mask = cv2.dilate(mask, kernel, iterations=1) # 팽창 (영역 확장 및 구멍 메우기)

        # 5. 윤곽선(Contour) 찾기
        # cv2.RETR_EXTERNAL: 가장 바깥쪽 윤곽선만 찾음
        # cv2.CHAIN_APPROX_SIMPLE: 윤곽선의 꼭짓점만 저장하여 메모리 절약
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 6. 찾은 각 윤곽선에 대해 처리
        output_image = image.copy() # 원본 이미지에 사각형을 그리기 위해 복사

        for cnt in contours:
            # 윤곽선 면적 계산 (너무 작은 노이즈 영역은 무시)
            area = cv2.contourArea(cnt)
            if area > 500: # 최소 면적 임계값 (조절 필요)
                # 윤곽선을 감싸는 최소한의 사각형 (ROI) 좌표 얻기
                x, y, w, h = cv2.boundingRect(cnt)
                
                # 사각형 그리기 (원본 이미지에)
                # cv2.rectangle(이미지, 시작점(x,y), 끝점(x+w, y+h), 색상(BGR), 두께)
                cv2.rectangle(output_image, (x, y), (x+w, y+h), (0, 255, 0), 2) # 초록색 사각형

                # 감지된 물체 위에 텍스트 표시 (선택 사항)
                cv2.putText(output_image, "Blue Object", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return output_image

    def stop_callback(self, msg):
        """
        'STOP' 토픽 메시지가 수신될 때 호출되는 콜백 함수.
        이 함수가 호출되면 현재 카메라 프레임을 저장합니다.
        """
        self.get_logger().info("STOP 신호 수신! 현재 카메라 프레임을 저장합니다.")
        self.save_current_frame()

    def save_current_frame(self):
        """
        현재 프레임을 파일로 저장하는 메서드.
        날짜별 폴더를 생성하고 그 안에 이미지를 저장합니다.
        """
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

            # 파일 이름에 시간 정보 포함
            timestamp = datetime.datetime.now().strftime("%H-%M-%S")
            filename = os.path.join(date_specific_dir, f"capture_{timestamp}.jpg")

            try:
                # 저장 시에는 ROI가 그려진 프레임(current_frame)을 저장
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
    node = TurtlebotObjectDetector()

    print("\n--- 터틀봇 객체 감지 및 캡처 노드 ---")
    print("노드가 실행 중입니다. 카메라 영상에 파란색 물체가 사각형으로 표시됩니다.")
    print("'/stop_signal' 토픽이 발행되면 현재 화면의 이미지가 저장됩니다.")
    print("ROS 2 터미널에서 'ros2 topic pub /stop_signal std_msgs/msg/Empty '{}'' 명령어를 실행하여 캡처를 트리거하세요.")
    print("노드를 종료하려면 Ctrl+C를 누르십시오.")

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        node.get_logger().info('노드 종료 요청 (Ctrl+C).')
    finally:
        node.destroy_node()
        cv2.destroyAllWindows() # OpenCV 윈도우 닫기
        rclpy.shutdown()

if __name__ == '__main__':
    main()