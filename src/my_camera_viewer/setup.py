from setuptools import find_packages, setup

package_name = 'my_camera_viewer'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
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
            'image_viewer = my_camera_viewer.image_viewer:main',
            'image_capture = my_camera_viewer.image_capture:main',
            'test = my_camera_viewer.test:main',
            'gazebo = my_camera_viewer.gazebo:main',
            'new = my_camera_viewer.new:main',
            'new2 = my_camera_viewer.new2:main',
        ],
    },
)
