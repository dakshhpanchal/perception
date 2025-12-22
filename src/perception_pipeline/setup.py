from setuptools import setup
import os
from glob import glob

package_name = 'perception_pipeline'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), 
         glob('launch/*.xml') + glob('launch/*.py')),
        ('share/' + package_name + '/rviz',
            glob('rviz/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='your_name',
    maintainer_email='your_email@example.com',
    description='Perception pipeline for white lane and pothole detection',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'white_filter = perception_pipeline.white_filter:main',
            'costmap_generator = perception_pipeline.costmap_generator:main',
            'visualizer = perception_pipeline.visualizer:main',
        ],
    },
)