import os

# Get the file path from user input
file_path = './results_detection/model_espnetv2_coco/s_2.0_sch_hybrid_im_300/20211118-230357/espnetv2_2.0_checkpoint.pth.tar'
#input("Enter the file path: ")

# Remove the "./" prefix from the file path
if file_path.startswith("./"):
    file_path = file_path[2:]

# Get the current working directory
current_dir = os.getcwd()

# Create the full file path
full_file_path = os.path.join(current_dir, file_path)

# Create the directory path
directory_path = os.path.dirname(full_file_path)

# Replace forward slashes with backslashes
directory_path = directory_path.replace("/", "\\")

# Add the Windows root directory and the EdgeNets path if they're missing
if not directory_path.startswith("D:\\ROCSAFE\\Repos\\EdgeNets"):
    directory_path = os.path.join("D:\\ROCSAFE\\Repos\\EdgeNets", directory_path)

# Check if the directory exists
if not os.path.exists(directory_path):
    # Create the directory if it doesn't exist
    os.makedirs(directory_path)
    print(f"Directory created: {directory_path}")
else:
    print(f"Directory already exists: {directory_path}")
