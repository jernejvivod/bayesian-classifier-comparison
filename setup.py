from setuptools import setup

setup(name='bayesian_classifier_comparison',
      version='0.2',
      description='Compare performance of classification algorithms using Bayesian statistics',
      url='https://github.com/jernejvivod/bayesian-classifier-comparison',
      author='Jernej Vivod',
      author_email='vivod.jernej@gmail.com',
      packages=['bayesian_classifier_comparison'],
      install_requires=[
          'numpy',
          'scipy',
          'sklearn',
          'julia',
      ],
      classifiers=[
          'Intended Audience :: Developers',
          'Intended Audience :: Information Technology',
          'Intended Audience :: Science/Research',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.4',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          'Topic :: Utilities'
      ],
      keywords=['data mining', 'machine learning', 'data analysis', 'artificial intelligence', 'data science', 'classification'],
      include_package_data=True,
      zip_safe=False)

