import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    package_share = get_package_share_directory('object_tracking')
    sam_node_launch = os.path.join(package_share, 'launch', 'sam_node.launch.py')

    venv_python = LaunchConfiguration('venv_python')
    mpl_config_dir = LaunchConfiguration('mpl_config_dir')
    use_sam = LaunchConfiguration('use_sam')
    model_mode = LaunchConfiguration('model_mode')
    search_angular_speed = LaunchConfiguration('search_angular_speed')
    image_topic = LaunchConfiguration('image_topic')
    input_reliability = LaunchConfiguration('input_reliability')
    use_depth_input = LaunchConfiguration('use_depth_input')
    depth_topic = LaunchConfiguration('depth_topic')
    prompt_topic = LaunchConfiguration('prompt_topic')
    cmd_vel_topic = LaunchConfiguration('cmd_vel_topic')
    target_pixel_topic = LaunchConfiguration('target_pixel_topic')
    target_mask_topic = LaunchConfiguration('target_mask_topic')
    goal_locked_topic = LaunchConfiguration('goal_locked_topic')
    image_out_topic = LaunchConfiguration('image_out_topic')
    enable_search_rotation = LaunchConfiguration('enable_search_rotation')
    depth_match_tolerance = LaunchConfiguration('depth_match_tolerance')
    target_publish_rate = LaunchConfiguration('target_publish_rate')
    continuous_frame_max_age = LaunchConfiguration('continuous_frame_max_age')
    publish_mask_in_continuous = LaunchConfiguration('publish_mask_in_continuous')

    continuous_tracker = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(sam_node_launch),
        launch_arguments={
            'venv_python': venv_python,
            'mpl_config_dir': mpl_config_dir,
            'use_sam': use_sam,
            'model_mode': model_mode,
            'search_angular_speed': search_angular_speed,
            'use_compressed_input': 'true',
            'input_reliability': input_reliability,
            'image_topic': image_topic,
            'prompt_topic': prompt_topic,
            'cmd_vel_topic': cmd_vel_topic,
            'target_pixel_topic': target_pixel_topic,
            'target_mask_topic': target_mask_topic,
            'goal_locked_topic': goal_locked_topic,
            'burst_complete_topic': '/tracker/burst_complete',
            'tracking_mode': 'continuous',
            'use_depth_input': use_depth_input,
            'depth_topic': depth_topic,
            'depth_match_tolerance': depth_match_tolerance,
            'target_publish_rate': target_publish_rate,
            'continuous_frame_max_age': continuous_frame_max_age,
            'publish_mask_in_continuous': publish_mask_in_continuous,
            'enable_search_rotation': enable_search_rotation,
            'image_out_topic': image_out_topic,
        }.items(),
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'venv_python',
            default_value='~/.venvs/ros-jazzy-ml/bin/python',
            description='Python interpreter inside the ML virtual environment.',
        ),
        DeclareLaunchArgument(
            'mpl_config_dir',
            default_value='/tmp/object_tracking-mpl',
            description='Writable directory for Matplotlib config/cache.',
        ),
        DeclareLaunchArgument(
            'use_sam',
            default_value='false',
            description='Legacy switch. Used only when model_mode:=auto.',
        ),
        DeclareLaunchArgument(
            'model_mode',
            default_value='auto',
            description='Segmentation backend: auto, clip, dino_mobilesam, or yoloe.',
        ),
        DeclareLaunchArgument(
            'search_angular_speed',
            default_value='0.5',
            description='Angular velocity used while searching for the target.',
        ),
        DeclareLaunchArgument(
            'image_topic',
            default_value='/tracker/color/image_raw/compressed',
            description='Continuous compressed RGB topic consumed by the laptop-side tracker.',
        ),
        DeclareLaunchArgument(
            'input_reliability',
            default_value='reliable',
            description='QoS reliability for incoming compressed RGB/depth over Wi-Fi. Reliable is safer for inter-machine transport.',
        ),
        DeclareLaunchArgument(
            'use_depth_input',
            default_value='true',
            description='Whether the laptop-side continuous tracker should also subscribe to depth so it can choose the nearest valid point on the segmented mask.',
        ),
        DeclareLaunchArgument(
            'depth_topic',
            default_value='/tracker/aligned_depth_to_color/image_raw',
            description='Raw depth topic consumed directly by the laptop-side continuous tracker.',
        ),
        DeclareLaunchArgument(
            'prompt_topic',
            default_value='/target_prompt',
            description='Incoming text prompt topic.',
        ),
        DeclareLaunchArgument(
            'cmd_vel_topic',
            default_value='/cmd_vel_tracker',
            description='Twist topic used for search rotation.',
        ),
        DeclareLaunchArgument(
            'target_pixel_topic',
            default_value='/target_pixel',
            description='Pixel target topic published by the tracker.',
        ),
        DeclareLaunchArgument(
            'target_mask_topic',
            default_value='/target_mask',
            description='Binary segmentation mask topic published by the tracker.',
        ),
        DeclareLaunchArgument(
            'goal_locked_topic',
            default_value='/target_goal_locked',
            description='Latched Bool topic used to pause tracking after a goal lock on the Pi.',
        ),
        DeclareLaunchArgument(
            'image_out_topic',
            default_value='/image_out',
            description='Debug image topic with visualization overlay.',
        ),
        DeclareLaunchArgument(
            'enable_search_rotation',
            default_value='false',
            description='Whether the laptop-side tracker is allowed to rotate the robot while searching.',
        ),
        DeclareLaunchArgument(
            'depth_match_tolerance',
            default_value='0.2',
            description='Maximum allowed timestamp mismatch in seconds between RGB and depth frames.',
        ),
        DeclareLaunchArgument(
            'target_publish_rate',
            default_value='3.0',
            description='Maximum continuous publication rate in Hz for /target_pixel and /target_mask.',
        ),
        DeclareLaunchArgument(
            'publish_mask_in_continuous',
            default_value='false',
            description='Whether to publish /target_mask alongside /target_pixel in continuous mode. Keep false to reduce laptop -> Raspberry Pi bandwidth.',
        ),
        DeclareLaunchArgument(
            'continuous_frame_max_age',
            default_value='2.0',
            description='Drop a cached frame if it remained unprocessed longer than this many seconds.',
        ),
        continuous_tracker,
    ])
