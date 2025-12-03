from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # 获取包路径
    pkg_share = get_package_share_directory('camera_publisher')
    config_file = os.path.join(pkg_share, 'config', 'camera_params.yaml')
    
    # Launch参数
    device_id_arg = DeclareLaunchArgument(
        'device_id',
        default_value='0',
        description='相机设备ID'
    )
    
    frame_rate_arg = DeclareLaunchArgument(
        'frame_rate',
        default_value='30',
        description='相机帧率'
    )
    
    # 相机节点
    camera_node = Node(
        package='camera_publisher',
        executable='camera_node',
        name='camera_publisher',
        output='screen',
        parameters=[
            config_file,
            {
                'device_id': LaunchConfiguration('device_id'),
                'frame_rate': LaunchConfiguration('frame_rate')
            }
        ]
    )
    
    return LaunchDescription([
        device_id_arg,
        frame_rate_arg,
        camera_node
    ])
