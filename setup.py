import os
import subprocess
from setuptools import setup, find_packages
from setuptools.command.install import install

import dtw_live


VERSION = dtw_live.__version__

ROOT = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(ROOT, 'README.md')) as f:
    README = f.read()

REQUIRES = ['numpy', 'scipy', 'scikit-learn', 'matplotlib']


class CModuleInstall(install):
    def run(self):
        c_module_path = './dtw_live/dtw_c'
        subprocess.check_call('make', cwd=c_module_path, shell=True)
        super().run()


setup(
    name='dtw_live',
    version=VERSION,
    description='Real-time recognition using Dynamic Time Warping',
    long_description=README,
    python_requires='>=3.6.9',
    packages=find_packages(),
    install_requires=REQUIRES,
    extras_require={'tests': ['pytest']},
    cmdclass={'install': CModuleInstall},
    package_data={'dtw_live.dtw_c': ['*.so']},
    url='https://code.engineering.queensu.ca/13jvt/dtw_live',
    author='Johann von Tiesenhausen',
    author_email='13jvt@queensu.ca'
)
