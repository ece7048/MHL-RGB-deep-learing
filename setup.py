from setuptools import setup
from setuptools import find_packages
#import pip


#pip.main(['install', 'git+https://www.github.com/keras-team/keras-contrib.git'])

setup(name='ph_pipeline',
      version='0.2',
      description='Deep Learning classification pipeline analysis in Python',
      url='',
      author='Michail Mamalakis',
      author_email='mmamalakis1@sheffield.ac.uk',
      license='GPL-3.0+',
      packages=['ph_pipeline','ph_pipeline.XAI','ph_pipeline.XAI.GradCam'],
      install_requires=[
          'tensorflow>=2.2.0',
          'tensorboard>=2.2.0',
          'tensorflow-gpu>=2.2.0',
          'fastprogress>=1.0.0',
	  'keras>=2.4.1',
          'seaborn',
	  'six',
	  'einops>=0.3.0',
	  'h5py>=2.10.0',
          'numpy>=1.15.4',
          'scipy>=1.1.0',
	  'matplotlib>=3.1.0',
	  'dicom',
	  'pydicom>=2.1.1',	
	  'opencv-python>=4.0.0.21',
	  'Pillow>=5.3.0',
	  'vtk>=8.1.1',	
          'future',
	  'rq-scheduler>=0.7.0',
	  'med2image',
	  'imageio',
          'gensim',
          'SimpleItk',
          'networkx',
          'sklearn',
	  'pypac',
	  'nibabel',
	   'np_utils',
	   'medpy',
	   "onednn-cpu-gomp==2022.0.1",
	   "scikit-image",
	    "nets",
      ],
	zip_safe=False

)


