import launch

from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    ld = LaunchDescription([
        Node(
            package='nav2_map_server',
            executable='map_server',
            output='screen',
            parameters=[
                launch.substitutions.LaunchConfiguration('map_config')
            ]
        ),
        Node(
            package='nav2_lifecycle_manager',
            executable='lifecycle_manager',
            name='lifecycle_manager',
            output='screen',
            emulate_tty=True,  # https://github.com/ros2/launch/issues/188
            parameters=[{'use_sim_time': True},
                        {'autostart': True},
                        {'node_names': ['map_server']}])
    ])

    return ld

if __name__ == '__main__':
    generate_launch_description()