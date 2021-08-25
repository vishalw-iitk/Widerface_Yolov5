'''To get the file size given the file path and user defined name
This file is not in use for this pipeline
'''

'''Importing libraries'''
import os
import numpy as np

class check_file_size:
    '''check_file_size_in_the_given_path'''
    def __init__(self, file_path, file_name_user_defined):
        self.file_path = file_path
        self.file_name_user_defined = file_name_user_defined
    def size(self):
        self.filesize = os.path.getsize(self.file_path)
        self.size_in_mb = np.round(self.filesize / (1024*1024), 2)
        self.statement =  self.file_name_user_defined + " : " + str(self.size_in_mb) + " MB"
