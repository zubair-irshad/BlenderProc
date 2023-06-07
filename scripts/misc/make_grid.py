import os
import random
from PIL import Image


dir = '/home/zubairirshad/Downloads/FRONT3D_render'

folder_name = '3dfront_2000_00'

# Set the path to the folder containing the images
folder_path = os.path.join(dir, folder_name, 'train/images')

# Get a list of all image file names in the folder
image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.endswith(('.png', '.jpg', '.jpeg'))]

# Randomly select 18 images from the folder
selected_images = random.sample(image_files, 18)

# # Calculate the aspect ratio for the cropped images
# aspect_ratio = 640 / 480

# # Calculate the width and height for each cropped image based on the aspect ratio
# crop_width = int(image_size * aspect_ratio)
# crop_height = image_size

crop_width = int(640/4)
crop_height = int(480/4)
# Create a new blank image grid with a size of 3 by 6
grid_width = 3
grid_height = 6
image_size = 200  # Adjust this value to set the size of individual images in the grid
border_size = 5  # Adjust this value to set the size of the white border
grid = Image.new('RGB', (grid_width * (crop_width + border_size), grid_height * (crop_height + border_size)), color='white')

# Iterate over each selected image and paste it into the grid
for i, image_file in enumerate(selected_images):
    # Open the image file
    image_path = os.path.join(folder_path, image_file)
    image = Image.open(image_path)

    # Resize the image to fit in the grid
    image.thumbnail((crop_width, crop_height))

    # Create a new image with the desired aspect ratio and white background
    new_image = Image.new('RGB', (crop_width, crop_height), color='white')
    x_offset = (crop_width - image.width) // 2
    y_offset = (crop_height - image.height) // 2
    new_image.paste(image, (x_offset, y_offset))

    # Calculate the position to paste the image in the grid
    x = (i % grid_width) * (crop_width + border_size)
    y = (i // grid_width) * (crop_height + border_size)

    # Paste the image into the grid
    grid.paste(new_image, (x, y))

# Save the image grid as a PNG file
save_name = folder_name + '_grid.png'
grid.save(save_name)
# # Get a list of all image file names in the folder
# image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.endswith(('.png', '.jpg', '.jpeg'))]

# # Randomly select 18 images from the folder
# selected_images = random.sample(image_files, 18)

# # Create a new blank image grid with a size of 3 by 6
# grid_width = 3
# grid_height = 6
# image_size = 200  # Adjust this value to set the size of individual images in the grid
# grid = Image.new('RGB', (grid_width * image_size, grid_height * image_size))

# # Iterate over each selected image and paste it into the grid
# for i, image_file in enumerate(selected_images):
#     # Open the image file
#     image_path = os.path.join(folder_path, image_file)
#     image = Image.open(image_path)

#     # Resize the image to fit in the grid
#     image = image.resize((image_size, image_size))

#     # Calculate the position to paste the image in the grid
#     x = (i % grid_width) * image_size
#     y = (i // grid_width) * image_size

#     # Paste the image into the grid
#     grid.paste(image, (x, y))

# # Save the image grid as a PNG file
# grid.save('image_grid.png')