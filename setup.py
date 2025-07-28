#!/usr/bin/env python3
"""
Setup script for Chronic Stress Digital Twin Framework

Authors: Valentina Ezcurra, Ancuta Margondai, et al.
Institution: University of Central Florida
Conference: MODSIM World 2025
"""

from setuptools import setup, find_packages
import os

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Read requirements
with open('requirements.txt') as f:
    requirements = f.read().splitlines()
    # Filter out comments and empty lines
    requirements = [req for req in requirements if not req.startswith('#') and req.strip()]

setup(
    name="chronic-stress-digital-twin",
    version="1.0.0",
    description="Engineering Mental Wellness: A Digital Twin for Chronic Stress Modeling and Real-Time Intervention",
    long_description=long_description,
    long_description_content_type="text/markdown",
    
    # Author information
    author="Valentina Ezcurra, Ancuta Margondai",
    author_email="Valentinaezcurrabattro@gmail.com, Ancuta.Margondai@ucf.edu",
    
    # Project URLs
    url="https://github.com/yourusername/chronic-stress-digital-twin",
    
    # Package information
    python_requires=">=3.8",
    install_requires=requirements,
    
    # Classification
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    
    # Keywords
    keywords=[
        "digital-twin", "chronic-stress", "mental-health", "adolescent-psychology", 
        "agent-based-modeling", "stress-intervention", "anxiety-modeling"
    ],
)
