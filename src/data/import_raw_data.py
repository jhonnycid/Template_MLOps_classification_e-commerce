import requests
import os
import logging
import shutil
from check_structure import check_existing_file, check_existing_folder


def import_raw_data(raw_data_relative_path, filenames, bucket_folder_url, local_image_path=None):
    """
    Import filenames from bucket_folder_url in raw_data_relative_path.
    If local_image_path is provided, use it as a fallback for images.
    """
    if check_existing_folder(raw_data_relative_path):
        os.makedirs(raw_data_relative_path)
        
    # Download all the files
    for filename in filenames:
        input_file = os.path.join(bucket_folder_url, filename)
        output_file = os.path.join(raw_data_relative_path, filename)
        if check_existing_file(output_file):
            object_url = input_file
            print(f"Downloading {input_file} as {os.path.basename(output_file)}")
            try:
                response = requests.get(object_url)
                if response.status_code == 200:
                    # Process the response content as needed
                    content = response.content  # Use response.content for binary files
                    with open(output_file, "wb") as file:
                        file.write(content)
                else:
                    print(f"Error accessing the object {input_file}: {response.status_code}")
            except Exception as e:
                print(f"Error downloading {input_file}: {str(e)}")

    # Process the image folders
    for img_folder in ["image_train", "image_test"]:
        img_folder_url = os.path.join(bucket_folder_url, f"{img_folder}/")
        img_local_path = os.path.join(raw_data_relative_path, f"{img_folder}/")
        local_img_source = None
        
        if local_image_path:
            local_img_source = os.path.join(local_image_path, img_folder)
        
        # Create the target directory if it doesn't exist
        if check_existing_folder(img_local_path):
            os.makedirs(img_local_path)
        else:
            # Check if the directory is empty
            if len(os.listdir(img_local_path)) > 0:
                print(f"Directory {img_local_path} already contains files, skipping...")
                continue
        
        # Try to download from AWS
        aws_success = False
        try:
            response = requests.get(img_folder_url, timeout=10)  # Add timeout to prevent long waits
            if response.status_code == 200:
                file_list = response.text.splitlines()
                if file_list:  # Check if we got a list of files
                    aws_success = True
                    print(f"Downloading images from AWS for {img_folder}...")
                    for img_url in file_list:
                        img_filename = os.path.basename(img_url)
                        output_file = os.path.join(img_local_path, img_filename)
                        if check_existing_file(output_file):
                            print(f"Downloading {img_url} as {img_filename}")
                            img_response = requests.get(img_url)
                            if img_response.status_code == 200:
                                with open(output_file, "wb") as img_file:
                                    img_file.write(img_response.content)
                            else:
                                print(f"Error downloading {img_url}: {img_response.status_code}")
            else:
                print(f"Error accessing the object list {img_folder_url}: {response.status_code}")
        except Exception as e:
            print(f"Error accessing AWS for {img_folder}: {str(e)}")
        
        # If AWS download failed or wasn't attempted, try to copy from local path
        if not aws_success and local_img_source and os.path.exists(local_img_source):
            print(f"Copying images from local directory {local_img_source} to {img_local_path}")
            
            # Count files to copy for progress indication
            files_to_copy = [f for f in os.listdir(local_img_source) if os.path.isfile(os.path.join(local_img_source, f))]
            total_files = len(files_to_copy)
            
            # Copy files with progress indication
            for i, file_name in enumerate(files_to_copy):
                if i % 100 == 0:  # Show progress every 100 files
                    print(f"Copying files: {i}/{total_files} ({i/total_files*100:.1f}%)")
                
                source_file = os.path.join(local_img_source, file_name)
                dest_file = os.path.join(img_local_path, file_name)
                
                if not os.path.exists(dest_file):  # Only copy if the file doesn't exist
                    shutil.copy2(source_file, dest_file)
            
            print(f"Completed copying {total_files} files to {img_local_path}")
        elif not aws_success and not (local_img_source and os.path.exists(local_img_source)):
            print(f"WARNING: Could not download or copy images for {img_folder}. AWS failed and local path {local_img_source} not available.")


def main(
    raw_data_relative_path="./data/raw",
    filenames=["X_test_update.csv", "X_train_update.csv", "Y_train_CVw08PX.csv"],
    bucket_folder_url="https://mlops-project-db.s3.eu-west-1.amazonaws.com/classification_e-commerce/",
    local_image_path="/Users/danhang/Documents/PROJET TRANSITION PRO/Formation IA/DataScientest/Projet Rakuten/archive/images/images"
):
    """
    Upload data from AWS s3 in ./data/raw.
    If AWS is not available, try to copy from local_image_path.
    """
    import_raw_data(raw_data_relative_path, filenames, bucket_folder_url, local_image_path)
    logger = logging.getLogger(__name__)
    logger.info("Making raw data set completed")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # You can override local_image_path by adding an argument here
    main()