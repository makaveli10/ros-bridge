import os
import launch
import launch_ros.actions
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    ld = launch.LaunchDescription([
        launch.actions.DeclareLaunchArgument(
            name='host',
            default_value='localhost',
            description='IP of the CARLA server'
        ),
        launch.actions.DeclareLaunchArgument(
            name='port',
            default_value='2000',
            description='TCP port of the CARLA server'
        ),
        launch.actions.DeclareLaunchArgument(
            name='websocket_port',
            default_value='9090',
            description='Web socket port'
        ),
        launch.actions.DeclareLaunchArgument(
            name='timeout',
            default_value='2',
            description='Time to wait for a successful connection to the CARLA server'
        ),
        launch.actions.DeclareLaunchArgument(
            name='passive',
            default_value='False',
            description='When enabled, the ROS bridge will take a backseat and another client must tick the world (only in synchronous mode)'
        ),
        launch.actions.DeclareLaunchArgument(
            name='synchronous_mode',
            default_value='True',
            description='Enable/disable synchronous mode. If enabled, the ROS bridge waits until the expected data is received for all sensors'
        ),
        launch.actions.DeclareLaunchArgument(
            name='synchronous_mode_wait_for_vehicle_control_command',
            default_value='False',
            description='When enabled, pauses the tick until a vehicle control is completed (only in synchronous mode)'
        ),
        launch.actions.DeclareLaunchArgument(
            name='fixed_delta_seconds',
            default_value='0.05',
            description='Simulation time (delta seconds) between simulation steps'
        ),
        launch.actions.DeclareLaunchArgument(
            name='town',
            default_value='Town01',
            description='Either use an available CARLA town (eg. "Town01") or an OpenDRIVE file (ending in .xodr)'
        ),
        launch.actions.DeclareLaunchArgument(
            name='register_all_sensors',
            default_value='True',
            description='Enable/disable the registration of all sensors. If disabled, only sensors spawned by the bridge are registered'
        ),
        # Let the bridge know that vehicles with these role names should be identified as 
        # ego vehicles.  Only the vehicles within this list are controllable from within 
        # ROS (the vehicle from CARLA is selected as ego_vehicle which has the attribute 'role_name' 
        # set to this value).Choose one of these role_names for you ego_vehicle or pass your own
        # custom list and use a role_name from the new list.
        launch.actions.DeclareLaunchArgument(
            name='ego_vehicle_role_name',
            default_value=["ego_vehicle", "hero0", "hero1", "hero2", "hero3", "hero4", "hero5", "hero6", "hero7", 
                           "hero8", "hero9", "hero10", "hero11", "hero12", "hero13", "hero14", "hero15", "hero16", 
                           "hero17", "hero18", "hero19"],
            description='Role names to identify ego vehicles. '
        ),
        launch.actions.DeclareLaunchArgument(
            name='objects_definition_file',
            default_value=os.path.join(get_package_share_directory(
                'carla_spawn_objects'), 'config', 'objects_no_ego.json')
        ),
        launch.actions.DeclareLaunchArgument(
            name='map_config',
            default_value=os.path.join(
                get_package_share_directory('carla_ros_bridge'), 
                'map_Town01.yaml'),
            description='Nav2 map server config yaml file path'
        ),
        launch_ros.actions.Node(
            package='carla_ros_bridge',
            executable='bridge',
            name='carla_ros_bridge',
            output='screen',
            emulate_tty='True',
            on_exit=launch.actions.Shutdown(),
            parameters=[
                {
                    'use_sim_time': True
                },
                {
                    'host': launch.substitutions.LaunchConfiguration('host')
                },
                {
                    'port': launch.substitutions.LaunchConfiguration('port')
                },
                {
                    'timeout': launch.substitutions.LaunchConfiguration('timeout')
                },
                {
                    'passive': launch.substitutions.LaunchConfiguration('passive')
                },
                {
                    'synchronous_mode': launch.substitutions.LaunchConfiguration('synchronous_mode')
                },
                {
                    'synchronous_mode_wait_for_vehicle_control_command': launch.substitutions.LaunchConfiguration('synchronous_mode_wait_for_vehicle_control_command')
                },
                {
                    'fixed_delta_seconds': launch.substitutions.LaunchConfiguration('fixed_delta_seconds')
                },
                {
                    'town': launch.substitutions.LaunchConfiguration('town')
                },
                {
                    'register_all_sensors': launch.substitutions.LaunchConfiguration('register_all_sensors')
                },
                {
                    'ego_vehicle_role_name': launch.substitutions.LaunchConfiguration('ego_vehicle_role_name')
                }
            ]
        ),
        launch.actions.IncludeLaunchDescription(
            launch.launch_description_sources.PythonLaunchDescriptionSource(
                os.path.join(get_package_share_directory(
                    'carla_spawn_objects'), 'carla_example_ego_vehicle.launch.py')
            ),
            launch_arguments={
                'host': launch.substitutions.LaunchConfiguration('host'),
                'port': launch.substitutions.LaunchConfiguration('port'),
                'timeout': launch.substitutions.LaunchConfiguration('timeout'),
                'vehicle_filter': launch.substitutions.LaunchConfiguration('vehicle_filter'),
                'role_name': launch.substitutions.LaunchConfiguration('role_name'),
                'spawn_point': launch.substitutions.LaunchConfiguration('spawn_point'),
                'objects_definition_file': launch.substitutions.LaunchConfiguration('objects_definition_file')
            }.items()
        ),
        launch.actions.IncludeLaunchDescription(
            launch.launch_description_sources.AnyLaunchDescriptionSource(
                os.path.join(os.path.join(get_package_share_directory(
                    'rosbridge_server'), 'launch'), 'rosbridge_websocket_launch.xml')
            ),
            launch_arguments={
                'port': launch.substitutions.LaunchConfiguration('websocket_port')
            }.items()
        ),
        launch.actions.IncludeLaunchDescription(
            launch.launch_description_sources.PythonLaunchDescriptionSource(
                os.path.join(get_package_share_directory(
                    'carla_ros_bridge'), 'carla_ros_bridge_map_server.launch.py')
            ),
            launch_arguments={
                'map_config':  launch.substitutions.LaunchConfiguration('map_config')
            }.items()
        ),
    ])
    return ld


if __name__ == '__main__':
    generate_launch_description()
