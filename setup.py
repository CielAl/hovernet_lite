from setuptools import setup
with open("requirements.txt", "r") as f:
    requirements = f.read().splitlines()
setup(
    name='hovernet_lite',
    version='',
    packages=['hovernet_lite'],
    url='',
    license='',
    author='CielAl',
    author_email='',
    description='',
    install_requires=requirements
)
