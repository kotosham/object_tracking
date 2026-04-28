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
    env['PYTHONUNBUFFERED'] = '1'

    cmd = [
        venv_python,
        '-m',
        'object_tracking.tracker_node',
        '--ros-args',
        '-p',
        f"use_sam:={LaunchConfiguration('use_sam').perform(context)}",
        '-p',
        f"search_angular_speed:={LaunchConfiguration('search_angular_speed').perform(context)}",
        '-r',
        f"/image_in:={LaunchConfiguration('image_topic').perform(context)}",
        '-r',
        f"/depth_camera/depth/image_raw:={LaunchConfiguration('depth_topic').perform(context)}",
        '-r',
        f"/camera/camera_info:={LaunchConfiguration('camera_info_topic').perform(context)}",
        '-r',
        f"/target_prompt:={LaunchConfiguration('prompt_topic').perform(context)}",
        '-r',
        f"/cmd_vel:={LaunchConfiguration('cmd_vel_topic').perform(context)}",
        '-r',
        f"/goal_pose:={LaunchConfiguration('goal_topic').perform(context)}",
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
            default_value='true',
            description='Use GroundingDINO + SAM if true, otherwise use CLIPSeg.',
        ),
        DeclareLaunchArgument(
            'search_angular_speed',
            default_value='0.5',
            description='Angular velocity used while searching for the target.',
        ),
        DeclareLaunchArgument(
            'image_topic',
            default_value='/camera/image_raw',
            description='RGB image topic.',
        ),
        DeclareLaunchArgument(
            'depth_topic',
            default_value='/depth_camera/depth/image_raw',
            description='Depth image topic.',
        ),
        DeclareLaunchArgument(
            'camera_info_topic',
            default_value='/camera/camera_info',
            description='Camera info topic.',
        ),
        DeclareLaunchArgument(
            'prompt_topic',
            default_value='/target_prompt',
            description='Incoming text prompt topic.',
        ),
        DeclareLaunchArgument(
            'cmd_vel_topic',
            default_value='/cmd_vel',
            description='Twist topic used for search rotation.',
        ),
        DeclareLaunchArgument(
            'goal_topic',
            default_value='/goal_pose',
            description='Goal pose topic published by the tracker.',
        ),
        DeclareLaunchArgument(
            'image_out_topic',
            default_value='/image_out',
            description='Debug image topic with visualization overlay.',
        ),
        OpaqueFunction(function=launch_setup),
    ])
