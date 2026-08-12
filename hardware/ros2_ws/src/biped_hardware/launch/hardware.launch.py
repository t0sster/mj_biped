"""Запуск ноды сбора данных с железа:  ros2 launch biped_hardware hardware.launch.py"""

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    hardware_node = Node(
        package='biped_hardware',
        executable='hardware_node',
        name='hardware_node',
        output='screen',
    )
    return LaunchDescription([hardware_node])
