from setuptools import find_packages, setup
from glob import glob

package_name = 'object_tracking'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/model_weights', glob('object_tracking/model_weights/*.pth')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='clv',
    maintainer_email='dnbabkov@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'tracker_node = object_tracking.tracker_node:main'
        ],
    },
)
