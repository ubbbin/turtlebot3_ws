import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    pkg_name = 'patrol_robot_original_pkg'
    pkg_share_directory = get_package_share_directory(pkg_name)

    # 1. Main Robot Controller Node
    main_controller_node = Node(
        package=pkg_name,
        executable='main_controller',
        name='main_robot_controller',
        output='screen',
        emulate_tty=True, # For colored output from ros2_logger
        # parameters=[{'param_name': 'param_value'}] # 필요한 경우 파라미터 추가
    )

    # 2. Patrol Controller Node
    patrol_controller_node = Node(
        package=pkg_name,
        executable='patrol_controller',
        name='patrol_robot_controller',
        output='screen',
        emulate_tty=True,
    )

    # 3. Object Aligner Node
    object_aligner_node = Node(
        package=pkg_name,
        executable='object_aligner',
        name='turtlebot_object_aligner',
        output='screen',
        emulate_tty=True,
    )

    # 4. Obstacle Circulate Capture Nodes (Avoider + Camera Capture)
    # 이 executable은 ObstacleCircleAvoider와 TurtlebotCameraCapture 두 노드를 MultiThreadedExecutor로 띄웁니다.
    circulate_capture_nodes = Node(
        package=pkg_name,
        executable='circulate_capture_nodes',
        name='obstacle_circulate_capture_group', # 그룹 이름으로 지정
        output='screen',
        emulate_tty=True,
    )

    return LaunchDescription([
        main_controller_node,
        patrol_controller_node,
        object_aligner_node,
        circulate_capture_nodes,
    ])