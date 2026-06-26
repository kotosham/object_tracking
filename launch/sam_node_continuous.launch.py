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
    nearest_depth_percentile = LaunchConfiguration('nearest_depth_percentile')
    nearest_depth_min_pixels = LaunchConfiguration('nearest_depth_min_pixels')
    target_publish_rate = LaunchConfiguration('target_publish_rate')
    continuous_frame_max_age = LaunchConfiguration('continuous_frame_max_age')
    continuous_rgb_stamp_max_age = LaunchConfiguration('continuous_rgb_stamp_max_age')
    publish_mask_in_continuous = LaunchConfiguration('publish_mask_in_continuous')
    clip_min_mask_area = LaunchConfiguration('clip_min_mask_area')
    dino_box_threshold = LaunchConfiguration('dino_box_threshold')
    dino_mobilesam_min_mask_area = LaunchConfiguration('dino_mobilesam_min_mask_area')
    florence2_model_id = LaunchConfiguration('florence2_model_id')
    florence2_task_prompt = LaunchConfiguration('florence2_task_prompt')
    florence2_max_new_tokens = LaunchConfiguration('florence2_max_new_tokens')
    florence2_num_beams = LaunchConfiguration('florence2_num_beams')
    clip_threshold = LaunchConfiguration('clip_threshold')
    florence2_min_mask_area = LaunchConfiguration('florence2_min_mask_area')
    yoloe_conf_threshold = LaunchConfiguration('yoloe_conf_threshold')
    yoloe_min_mask_area = LaunchConfiguration('yoloe_min_mask_area')

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
            'nearest_depth_percentile': nearest_depth_percentile,
            'nearest_depth_min_pixels': nearest_depth_min_pixels,
            'target_publish_rate': target_publish_rate,
            'continuous_frame_max_age': continuous_frame_max_age,
            'continuous_rgb_stamp_max_age': continuous_rgb_stamp_max_age,
            'publish_mask_in_continuous': publish_mask_in_continuous,
            'clip_min_mask_area': clip_min_mask_area,
            'dino_box_threshold': dino_box_threshold,
            'dino_mobilesam_min_mask_area': dino_mobilesam_min_mask_area,
            'florence2_model_id': florence2_model_id,
            'florence2_task_prompt': florence2_task_prompt,
            'florence2_max_new_tokens': florence2_max_new_tokens,
            'florence2_num_beams': florence2_num_beams,
            'clip_threshold': clip_threshold,
            'florence2_min_mask_area': florence2_min_mask_area,
            'yoloe_conf_threshold': yoloe_conf_threshold,
            'yoloe_min_mask_area': yoloe_min_mask_area,
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
            description='Segmentation backend: auto, clip, dino_mobilesam, florence2, or yoloe.',
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
            'nearest_depth_percentile',
            default_value='5.0',
            description='Depth percentile used instead of a raw minimum when selecting the nearest target point on a mask.',
        ),
        DeclareLaunchArgument(
            'nearest_depth_min_pixels',
            default_value='3',
            description='Minimum number of pixels required in the nearest-depth band before publishing a continuous target.',
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
            'florence2_model_id',
            default_value='microsoft/Florence-2-base-ft',
            description='Florence-2 model id or local snapshot directory name used for segmentation mode.',
        ),
        DeclareLaunchArgument(
            'florence2_task_prompt',
            default_value='<REFERRING_EXPRESSION_SEGMENTATION>',
            description='Florence-2 task token used for text-guided segmentation.',
        ),
        DeclareLaunchArgument(
            'florence2_max_new_tokens',
            default_value='1024',
            description='Maximum generated tokens for Florence-2 decoding.',
        ),
        DeclareLaunchArgument(
            'florence2_num_beams',
            default_value='3',
            description='Beam count for Florence-2 generation.',
        ),
        DeclareLaunchArgument(
            'clip_threshold',
            default_value='0.70',
            description='CLIPSeg mask probability threshold. Lower values make CLIPSeg less strict.',
        ),
        DeclareLaunchArgument(
            'clip_min_mask_area',
            default_value='100',
            description='Minimum accepted CLIPSeg mask area in pixels.',
        ),
        DeclareLaunchArgument(
            'dino_box_threshold',
            default_value='0.50',
            description='GroundingDINO confidence threshold before MobileSAM segmentation.',
        ),
        DeclareLaunchArgument(
            'dino_mobilesam_min_mask_area',
            default_value='100',
            description='Minimum accepted GroundingDINO + MobileSAM mask area in pixels.',
        ),
        DeclareLaunchArgument(
            'florence2_min_mask_area',
            default_value='100',
            description='Minimum accepted Florence-2 mask area in pixels.',
        ),
        DeclareLaunchArgument(
            'yoloe_conf_threshold',
            default_value='0.12',
            description='YOLOE-only confidence threshold.',
        ),
        DeclareLaunchArgument(
            'yoloe_min_mask_area',
            default_value='100',
            description='YOLOE-only minimum accepted mask area in pixels.',
        ),
        DeclareLaunchArgument(
            'continuous_frame_max_age',
            default_value='2.0',
            description='Drop a cached frame if it remained unprocessed longer than this many seconds.',
        ),
        DeclareLaunchArgument(
            'continuous_rgb_stamp_max_age',
            default_value='1.0',
            description='Drop a continuous RGB frame if its ROS header stamp is older than this many seconds at arrival/inference time.',
        ),
        continuous_tracker,
    ])
