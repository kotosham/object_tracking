import os

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
        f"target_publish_rate:={LaunchConfiguration('target_publish_rate').perform(context)}",
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
            description='Segmentation backend: auto, clip, dino_mobilesam, or yoloe.',
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
            'target_publish_rate',
            default_value='3.0',
            description='Maximum rate in Hz for publishing /target_pixel and /target_mask in continuous mode. Set <=0 for no limit.',
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
