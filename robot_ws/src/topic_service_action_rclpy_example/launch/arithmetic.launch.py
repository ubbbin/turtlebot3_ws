from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription(
        [
            Node(
                package='topic_service_action_rclpy_example',
                executable='argument', output='screen'),
            Node(
                package='topic_service_action_rclpy_example',
                executable='calculator', output='screen'),
        ]
    )
