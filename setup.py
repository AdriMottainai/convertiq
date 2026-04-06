from setuptools import find_packages
from setuptools import setup

with open("requirements_prod.txt") as f:
    content = f.readlines()
requirements = [x.strip() for x in content]

setup(name='convertiq',
      description="E-commerce purchase prediction using LightGBM",
      author="Adrien Roggero",
      install_requires=requirements,
      packages=find_packages()
)
