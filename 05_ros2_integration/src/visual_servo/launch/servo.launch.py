from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    pkg_share = get_package_share_directory('visual_servo')
    config_file = os.path.join(pkg_share, 'config', 'servo_params.yaml')
    
    servo_node = Node(
        package='visual_servo',
        executable='servo_node',
        name='visual_servo',
        output='screen',
        parameters=[config_file]
    )
    
    depth_estimator = Node(
        package='visual_servo',
        executable='depth_estimator',
        name='depth_estimator',
        output='screen',
        parameters=[config_file]
    )
    
    return LaunchDescription([
        servo_node,
        depth_estimator
    ])
