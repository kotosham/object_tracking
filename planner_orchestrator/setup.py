from setuptools import find_packages, setup

package_name = 'planner_orchestrator'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        # vlm.env.example ставится в share намеренно: консоль оператора создаёт
        # vlm.env из этого шаблона, чтобы не потерять документацию про формат
        # VLM_BASE_URL для Qwen/OpenAI/локального vLLM. Без установки шаблон
        # существует только в исходниках и в контейнере не находится.
        ('share/' + package_name, ['package.xml', 'vlm.env.example']),
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
