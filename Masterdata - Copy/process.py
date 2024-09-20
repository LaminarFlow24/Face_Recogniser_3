import os

def find_folders_with_image_count(directory):
    # Iterate over all sub-folders inside the specified directory
    for root, dirs, files in os.walk(directory):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            # Get all files in the sub-folder
            jpg_images = [file for file in os.listdir(dir_path) if file.lower().endswith(('.jpg', '.jpeg'))]
            image_count = len(jpg_images)
            
            # If there are no jpg/jpeg images, print the folder name
            if image_count == 0:
                print(f"Folder '{dir_name}' in path '{dir_path}' does not contain any jpg/jpeg images.")
            # If there are fewer than 3 jpg/jpeg images, print the folder name and count
            elif image_count < 3:
                print(f"Folder '{dir_name}' in path '{dir_path}' has less than 3 jpg/jpeg images. Count: {image_count}")

# Specify the directory to search
directory = '/path/to/your/directory'
find_folders_with_image_count(directory)
print("hello")