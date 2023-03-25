import os
import zipfile
import sys

def extract_zip_files(directory):
    for item in os.listdir(directory):
        full_path = os.path.join(directory, item)
        if zipfile.is_zipfile(full_path):
            with zipfile.ZipFile(full_path, 'r') as zip_ref:
                extract_directory = os.path.splitext(item)[0]
                extract_path = os.path.join(directory, extract_directory)
                os.makedirs(extract_path, exist_ok=True)
                zip_ref.extractall(extract_path)
                #os.remove(full_path)
print('hello')
root_directory = sys.argv[1]
print(root_directory)
extract_zip_files(root_directory)