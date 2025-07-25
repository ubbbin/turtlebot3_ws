from setuptools import find_packages, setup

package_name = 'patrol_robot_original_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/launch.py']), # 런치 파일 경로 추가 확인
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='yubin',
    maintainer_email='yubin@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'main_controller = patrol_robot_original_pkg.main_robot_controller:main',
            'patrol_controller = patrol_robot_original_pkg.patrol_controller_node:main',
            'object_aligner = patrol_robot_original_pkg.obstacle_detector_aligner_node:main',
            'circulate_capture_nodes = patrol_robot_original_pkg.obstacle_circulate_capture_node:main',
            'carmera_square = patrol_robot_original_pkg.carmera_square:main',
            'squrae_bot = patrol_robot_original_pkg.squrae_bot:main',

        ],
    },
)
