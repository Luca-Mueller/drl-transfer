from setuptools import setup

setup(name='torch_agents',
      version='0.1',
      description='RL Agents for PyTorch',
      url='https://github.com/Luca-Mueller/drl-transfer',
      author='Luca Mueller',
      license='MIT',
      packages=['torch_agents'],
      install_requires=['torch', 'gym', 'numpy', 'colorama', 'scipy', 'matplotlib'],
      zip_safe=False)
