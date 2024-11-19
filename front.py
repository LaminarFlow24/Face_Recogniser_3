import os
import shutil

def create_new_folder_with_subset_images(source_dir, destination_dir):
    """
    Creates a new folder structure with images from the 10th to 15th position in alphabetical order.
    """
    # Ensure the destination directory exists
    os.makedirs(destination_dir, exist_ok=True)
    
    for subfolder in os.listdir(source_dir):
        subfolder_path = os.path.join(source_dir, subfolder)
        if os.path.isdir(subfolder_path):  # Check if it is a directory
            # Get list of files in alphabetical order
            files = sorted(os.listdir(subfolder_path))
            
            # Select files from 10th to 15th (indices 9 to 14 in zero-based indexing)
            selected_files = files[9:15]
            
            # Create the corresponding subfolder in the destination directory
            new_subfolder_path = os.path.join(destination_dir, subfolder)
            os.makedirs(new_subfolder_path, exist_ok=True)
            
            for file in selected_files:
                source_file_path = os.path.join(subfolder_path, file)
                dest_file_path = os.path.join(new_subfolder_path, file)
                # Copy file to the new directory
                shutil.copy2(source_file_path, dest_file_path)
    
    print(f"Subset of images copied to {destination_dir}")

# Example usage
source_folder = "Output_folder"
destination_folder = "front_output_folder"
create_new_folder_with_subset_images(source_folder, destination_folder)
