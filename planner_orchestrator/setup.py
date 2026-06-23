from setuptools import find_packages, setup

package_name = 'planner_orchestrator'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='clv',
    maintainer_email='dnbabkov@gmail.com',
    description='Edge-side HTTP client to an external OpenAI-compatible VLM API (Phase 1.6 scaffold).',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'orchestrator_node = planner_orchestrator.orchestrator_node:main',
        ],
    },
)
