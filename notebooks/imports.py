'''
Allow imports from parent directory

Yuck, but see: https://stackoverflow.com/questions/34478398
'''

import os
import sys

module_path = os.path.abspath(os.path.join('..'))

if module_path not in sys.path:
    sys.path.append(module_path)

