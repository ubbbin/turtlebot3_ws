# #!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge # cv_bridge 임포트
import cv2 # OpenCV 임포트

class ImageViewer(Node):
    def __init__(self):
        super().__init__('image_viewer_node')
        self.get_logger().info("Image Viewer Node has been started.")

        # CvBridge 객체 생성
        self.bridge = CvBridge()

        # 이미지 토픽 구독자 생성
        # 터틀봇 카메라 영상 토픽 이름은 일반적으로 /camera/image_raw 또는 /usb_cam/image_raw 등입니다.
        # 본인의 로봇에서 발행하는 정확한 토픽 이름으로 변경해야 합니다.
        self.subscription = self.create_subscription(
            CompressedImage,
            'camera/image_raw/compressed',  # <- 이 부분을 로봇의 실제 카메라 토픽 이름으로 변경하세요!
            self.image_callback,
            10) # QoS depth

        self.subscription  # prevent unused variable warning

    def image_callback(self, msg):
        try:
            # ROS Image 메시지를 OpenCV 이미지(NumPy 배열)로 변환
            # 'bgr8'은 OpenCV에서 흔히 사용하는 BGR 채널 순서의 8비트 이미지 포맷입니다.
            cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")
            return

        # OpenCV로 이미지 화면에 표시
        cv2.imshow("Turtlebot3 Camera Feed", cv_image)
        # 키 입력을 1ms 기다리면서 화면을 업데이트 (필수)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args) # ROS2 초기화
    image_viewer = ImageViewer() # 노드 인스턴스 생성
    try:
        rclpy.spin(image_viewer) # 노드 실행 (콜백 함수를 계속 실행)
    except KeyboardInterrupt:
        # 노드 종료 시 자원 해제
        # rclpy.shutdown()
        image_viewer.destroy_node()
        cv2.destroyAllWindows() # OpenCV 윈도우 닫기 (필수)

if __name__ == '__main__':
    main()
