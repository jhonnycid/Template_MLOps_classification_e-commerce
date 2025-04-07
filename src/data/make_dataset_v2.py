import os
from check_structure import  check_existing_folder
import shutil

source_path = './data/raw'
destination_path = './data/preprocessed'

if os.path.exists(destination_path) is True:    
    shutil.rmtree(destination_path)
    shutil.move(source_path,destination_path)
    print("Dataset crée")
else:
    shutil.move(source_path,destination_path)
    print("Dataset crée")