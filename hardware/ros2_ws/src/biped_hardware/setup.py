from setuptools import find_packages, setup

package_name = 'biped_hardware'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/hardware.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ilya',
    maintainer_email='ilya200453il@gmail.com',
    description='Нода Raspberry Pi: моторы Damiao по CAN и IMU HWT906 по UART.',
    license='Apache 2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'hardware_node = biped_hardware.hardware_node:main',
            'imu_configure = biped_hardware.imu_configure:main',
        ],
    },
)
