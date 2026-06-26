import os
import sys

from ament_index_python.packages import get_package_prefix
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.substitutions import LaunchConfiguration


def launch_setup(context, *args, **kwargs):
    venv_python = os.path.expanduser(LaunchConfiguration('venv_python').perform(context))
    if not os.path.isfile(venv_python):
        raise RuntimeError(
            f"Python interpreter from venv was not found: {venv_python}. "
            "Create the ros-jazzy-ml environment or pass venv_python:=/path/to/python."
        )

    mpl_config_dir = os.path.expanduser(LaunchConfiguration('mpl_config_dir').perform(context))
    os.makedirs(mpl_config_dir, exist_ok=True)

    env = dict(os.environ)
    env['MPLCONFIGDIR'] = mpl_config_dir
    env.setdefault('HF_HOME', os.path.expanduser('~/.cache/huggingface'))
    env.setdefault('HF_HUB_DISABLE_PROGRESS_BARS', '1')
    env.setdefault('TRANSFORMERS_VERBOSITY', 'error')
    env['PYTHONUNBUFFERED'] = '1'
    package_prefix = get_package_prefix('object_tracking')
    package_site_packages = os.path.join(
        package_prefix,
        'lib',
        f'python{sys.version_info.major}.{sys.version_info.minor}',
        'site-packages',
    )
    env['PYTHONPATH'] = os.pathsep.join(
        path for path in [package_site_packages, env.get('PYTHONPATH', '')] if path
    )

    cmd = [
        venv_python,
        '-m',
        'object_tracking.rgb_tracker_node',
        '--ros-args',
        '-p',
        f"use_sam:={LaunchConfiguration('use_sam').perform(context)}",
        '-p',
        f"model_mode:={LaunchConfiguration('model_mode').perform(context)}",
        '-p',
        f"search_angular_speed:={LaunchConfiguration('search_angular_speed').perform(context)}",
        '-p',
        f"use_compressed_input:={LaunchConfiguration('use_compressed_input').perform(context)}",
        '-p',
        f"input_reliability:={LaunchConfiguration('input_reliability').perform(context)}",
        '-p',
        f"goal_locked_topic:={LaunchConfiguration('goal_locked_topic').perform(context)}",
        '-p',
        f"enable_search_rotation:={LaunchConfiguration('enable_search_rotation').perform(context)}",
        '-p',
        f"burst_quiet_period:={LaunchConfiguration('burst_quiet_period').perform(context)}",
        '-p',
        f"burst_complete_topic:={LaunchConfiguration('burst_complete_topic').perform(context)}",
        '-p',
        f"tracking_mode:={LaunchConfiguration('tracking_mode').perform(context)}",
        '-p',
        f"use_depth_input:={LaunchConfiguration('use_depth_input').perform(context)}",
        '-p',
        f"depth_topic:={LaunchConfiguration('depth_topic').perform(context)}",
        '-p',
        f"depth_match_tolerance:={LaunchConfiguration('depth_match_tolerance').perform(context)}",
        '-p',
        f"nearest_depth_percentile:={LaunchConfiguration('nearest_depth_percentile').perform(context)}",
        '-p',
        f"nearest_depth_min_pixels:={LaunchConfiguration('nearest_depth_min_pixels').perform(context)}",
        '-p',
        f"target_publish_rate:={LaunchConfiguration('target_publish_rate').perform(context)}",
        '-p',
        f"continuous_frame_max_age:={LaunchConfiguration('continuous_frame_max_age').perform(context)}",
        '-p',
        f"continuous_rgb_stamp_max_age:={LaunchConfiguration('continuous_rgb_stamp_max_age').perform(context)}",
        '-p',
        f"publish_mask_in_continuous:={LaunchConfiguration('publish_mask_in_continuous').perform(context)}",
        '-p',
        f"florence2_model_id:={LaunchConfiguration('florence2_model_id').perform(context)}",
        '-p',
        f"florence2_task_prompt:={LaunchConfiguration('florence2_task_prompt').perform(context)}",
        '-p',
        f"florence2_max_new_tokens:={LaunchConfiguration('florence2_max_new_tokens').perform(context)}",
        '-p',
        f"florence2_num_beams:={LaunchConfiguration('florence2_num_beams').perform(context)}",
        '-p',
        f"clip_threshold:={LaunchConfiguration('clip_threshold').perform(context)}",
        '-p',
        f"clip_min_mask_area:={LaunchConfiguration('clip_min_mask_area').perform(context)}",
        '-p',
        f"dino_box_threshold:={LaunchConfiguration('dino_box_threshold').perform(context)}",
        '-p',
        f"dino_mobilesam_min_mask_area:={LaunchConfiguration('dino_mobilesam_min_mask_area').perform(context)}",
        '-p',
        f"florence2_min_mask_area:={LaunchConfiguration('florence2_min_mask_area').perform(context)}",
        '-p',
        f"yoloe_conf_threshold:={LaunchConfiguration('yoloe_conf_threshold').perform(context)}",
        '-p',
        f"yoloe_min_mask_area:={LaunchConfiguration('yoloe_min_mask_area').perform(context)}",
        '-r',
        f"/image_in:={LaunchConfiguration('image_topic').perform(context)}",
        '-r',
        f"/target_prompt:={LaunchConfiguration('prompt_topic').perform(context)}",
        '-r',
        f"/cmd_vel:={LaunchConfiguration('cmd_vel_topic').perform(context)}",
        '-r',
        f"/target_pixel:={LaunchConfiguration('target_pixel_topic').perform(context)}",
        '-r',
        f"/target_mask:={LaunchConfiguration('target_mask_topic').perform(context)}",
        '-r',
        f"/image_out:={LaunchConfiguration('image_out_topic').perform(context)}",
    ]

    return [
        ExecuteProcess(
            cmd=cmd,
            env=env,
            output='screen',
        )
    ]


def generate_launch_description():
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
            'use_compressed_input',
            default_value='true',
            description='Subscribe to sensor_msgs/CompressedImage instead of raw Image.',
        ),
        DeclareLaunchArgument(
            'input_reliability',
            default_value='best_effort',
            description='QoS reliability used for incoming RGB/depth subscriptions: best_effort or reliable.',
        ),
        DeclareLaunchArgument(
            'image_topic',
            default_value='/tracker/color/image/compressed',
            description='RGB image topic or compressed RGB transport topic.',
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
            description='Binary segmentation mask topic for the selected target published by the tracker.',
        ),
        DeclareLaunchArgument(
            'goal_locked_topic',
            default_value='/target_goal_locked',
            description='Latched Bool topic used to pause laptop-side tracking once a goal is locked on the Pi.',
        ),
        DeclareLaunchArgument(
            'burst_complete_topic',
            default_value='/tracker/burst_complete',
            description='Topic used to signal the explicit end of an RGB burst from the Raspberry Pi.',
        ),
        DeclareLaunchArgument(
            'tracking_mode',
            default_value='burst',
            description='Tracker mode: burst or continuous.',
        ),
        DeclareLaunchArgument(
            'use_depth_input',
            default_value='false',
            description='Subscribe to a depth Image topic and match it to incoming RGB frames by timestamp.',
        ),
        DeclareLaunchArgument(
            'depth_topic',
            default_value='/camera/camera/aligned_depth_to_color/image_raw',
            description='Depth Image topic used only when use_depth_input:=true.',
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
            description='Maximum rate in Hz for publishing /target_pixel and /target_mask in continuous mode. Set <=0 for no limit.',
        ),
        DeclareLaunchArgument(
            'continuous_frame_max_age',
            default_value='2.0',
            description='Drop a cached frame in continuous mode if it sat unprocessed longer than this many seconds. Set <=0 to disable.',
        ),
        DeclareLaunchArgument(
            'continuous_rgb_stamp_max_age',
            default_value='0.0',
            description='Drop a continuous RGB frame if its ROS header stamp is older than this many seconds at arrival/inference time. Set <=0 to disable.',
        ),
        DeclareLaunchArgument(
            'publish_mask_in_continuous',
            default_value='false',
            description='Whether to publish /target_mask alongside /target_pixel in continuous mode.',
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
            description='YOLOE-only minimum accepted mask area in pixels. Defaults to the previous hard-coded value.',
        ),
        DeclareLaunchArgument(
            'enable_search_rotation',
            default_value='false',
            description='Whether the laptop-side tracker is allowed to rotate the robot while searching.',
        ),
        DeclareLaunchArgument(
            'burst_quiet_period',
            default_value='2.0',
            description='Fallback timeout in seconds used only if the explicit burst-complete signal is missing or incomplete.',
        ),
        DeclareLaunchArgument(
            'image_out_topic',
            default_value='/image_out',
            description='Debug image topic with visualization overlay.',
        ),
        OpaqueFunction(function=launch_setup),
    ])
