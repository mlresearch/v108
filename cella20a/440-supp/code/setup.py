from setuptools import setup, find_packages
from codecs import open
from os import path
import sys

here = path.abspath(path.dirname(__file__))
name = 'adaptiveRank'

# Get the long description from the README file
requires_list = []
with open(path.join(here, 'requirements.txt'), encoding = 'utf-8') as f:
	for line in f:
		requires_list.append(str(line))

sys.path.insert(0, path.join(path.dirname(__file__),name))
