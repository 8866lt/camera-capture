from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    pkg_share = get_package_share_directory('object_detector')
    config_file = os.path.join(pkg_share, 'config', 'detector_params.yaml')
    
    detector_node = Node(
        package='object_detector',
        executable='detector_node',
        name='object_detector',
        output='screen',
        parameters=[config_file]
    )
    
    return LaunchDescription([
        detector_node
    ])
