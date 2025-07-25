import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from rclpy.qos import QoSProfile
import termios
import tty
import os
import select
import sys

class Move_turtle(Node):
  def __init__(self):
    super().__init__('move_key_turtle')
    self.qos_profile = QoSProfile(depth = 10)
    self.move_key_turtle = self.create_publisher(Twist, '/cmd_vel', self.qos_profile)
    self.velocity = 0.0
    self.angular = 0.0
    self.settings = termios.tcgetattr(sys.stdin)

  def turtle_key_move(self):
    count = 0
    msg = Twist()
    print("input wasd")
    while True:
        input_key = self.get_key(self.settings)
        if input_key in ['w','W']:
            count += 1
            self.velocity += 0.1
            self.get_logger().info(f'Published mesage: {msg.linear}, {msg.angular}')
        elif input_key in ['s','S']:
            self.velocity = 0.0
            self.angular = 0.0
            count += 1
            self.get_logger().info(f'Published mesage: {msg.linear}, {msg.angular}')
        elif input_key in ['x','X']:
            self.velocity += -0.1
            count += 1
            self.get_logger().info(f'Published mesage: {msg.linear}, {msg.angular}')
        elif input_key in ['a','A']:
            self.angular += 0.1
            count += 1
            self.get_logger().info(f'Published mesage: {msg.linear}, {msg.angular}')
        elif input_key in ['d','D']:
            self.angular -= 0.1
            count += 1
            self.get_logger().info(f'Published mesage: {msg.linear}, {msg.angular}')
        elif input_key == '\x03':
            break
          
        msg.linear.x = self.velocity
        msg.linear.y = 0.0
        msg.linear.z = 0.0

        msg.angular.x = 0.0
        msg.angular.y = 0.0
        msg.angular.z = self.angular
        self.move_key_turtle.publish(msg)
        if count == 20:
            print("input wasd")

  def get_key(self, settings):
    if os.name == 'nt':
        return msvcrt.getch().decode('utf-8')
    tty.setraw(sys.stdin.fileno())
    rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
    if rlist:
        key = sys.stdin.read(1)
    else:
        key = ''

    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key

def main(args=None):
  rclpy.init(args=args)
  node = Move_turtle()
  try:
    node.turtle_key_move()
  except KeyboardInterrupt:
    node.get_logger().info('Keyboard interrupt!!!!')
  finally:
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main':
  main()