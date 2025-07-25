import rclpy as rp
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from my_move_turtle_pkg.move_turtle_pub import Move_turtle
from my_move_turtle_pkg.my_subscriber import TurtlesimSubscriber


def main(args=None):
    rp.init()

    sub = Move_turtle()
    pub = TurtlesimSubscriber()

    executor = MultiThreadedExecutor()

    executor.add_node(sub)
    executor.add_node(pub)

    try:
        executor.spin()

    finally:
        executor.shutdown()
        sub.destroy_node()
        pub.destroy_node()
        rp.shutdown()


if __name__ == '__main__':
    main()
