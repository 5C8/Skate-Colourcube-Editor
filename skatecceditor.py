import numpy as np
from tkinter import Tk, filedialog, Button, Canvas, Scale, OptionMenu, StringVar, HORIZONTAL, Frame, Label
from PIL import Image, ImageTk
import os
from concurrent.futures import ThreadPoolExecutor
import threading
import ctypes
import time

# Debounce implementation
class Debouncer:
    def __init__(self, func, delay):
        self.func = func
        self.delay = delay
        self.timer = None

    def call(self, *args, **kwargs):
        if self.timer:
            self.timer.cancel()  # Cancel any pending call
        self.timer = threading.Timer(self.delay, self.func, args, kwargs)
        self.timer.start()

# Global debounce instances
debounced_apply_transformation = None

def apply_transformation_debounced(*args):
    global debounced_apply_transformation
    if debounced_apply_transformation:
        debounced_apply_transformation.call()

last_loaded_lut = None
current_lut_file_path = None
lut_files = []
image_files = []
current_lut_index = -1
current_image_index = -1

platform = 'PS3'
platform_swap = False
zoom_level = 1.0  # Start with a default zoom level
x_offset = 0  # Initial horizontal offset
y_offset = 0  # Initial vertical offset

# Size of the reduced 32x32x32 color cube
size = 32
scale_factor = 255 / (size - 1)
num_coords = size ** 3

def load_lut_files_from_directory(directory):
    global lut_files, current_lut_index
    lut_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(('.rgb', '.cube'))]
    if lut_files:
        if current_lut_file_path in lut_files:
            current_lut_index = lut_files.index(current_lut_file_path)
        else:
            # If the current LUT file is not in the list, default to the first file
            current_lut_index = 0
    else:
        print("No .rgb or .cube files found in the directory.")
        return []

def load_image_files_from_directory(directory):
    global image_files, current_image_index
    image_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(('.png', '.jpg', '.jpeg'))]
    if image_files:
        if current_lut_file_path in image_files:
            current_image_index = image_files.index(current_lut_file_path)
        else:
            # If the current LUT file is not in the list, default to the first file
            current_image_index = 0
    else:
        print("No .rgb or .cube files found in the directory.")
        return []

def on_drag_start(event):
    global x_offset, y_offset, drag_start_x, drag_start_y
    drag_start_x = event.x
    drag_start_y = event.y

def on_drag_motion(event):
    global x_offset, y_offset, drag_start_x, drag_start_y

    if 'drag_start_x' not in globals() or 'drag_start_y' not in globals():
        # Initialize the drag starting position if not already initialized
        drag_start_x = event.x
        drag_start_y = event.y
        return  # Exit early if the drag has not started yet
    
    dx = event.x - drag_start_x
    dy = event.y - drag_start_y
    x_offset += dx
    y_offset += dy
    drag_start_x = event.x
    drag_start_y = event.y
    update_image_position()

def update_image_position():
    global x_offset, y_offset, zoom_level, blended_image
    canvas.delete("all")  # Clear existing image
    
    if blended_image is not None:
        # Ensure the updated transformed image is used
        show_image(blended_image)  # Display the transformed image with updated position and zoom
    else:
        if image is not None:
            show_image(image)
        else:
            print("No image available.")  # Debugging statement

def on_mouse_wheel(event):
    global zoom_level, x_offset, y_offset
    # Zoom in if the wheel is scrolled up, zoom out if scrolled down
    if event.delta > 0:
        zoom_level *= 1.1  # Zoom in (increase zoom level)
    else:
        zoom_level /= 1.1  # Zoom out (decrease zoom level)
    
    # Limit zoom level to a reasonable range
    zoom_level = max(0.1, min(zoom_level, 5.0))

    # Adjust the image size and position accordingly
    update_image_position()

# Precompute mapping
def precompute_cube_order(size):
    default_map = np.zeros((size, size, size), dtype=np.int32)
    for x in range(size):
        for y in range(size):
            for z in range(size):
                index = x * size**2 + y * size + z
                default_map[x, y, z] = index
    return default_map

# Precompute Z-curve mapping
def precompute_z_curve(size):
    bits = size.bit_length() - 1
    z_curve_map = np.zeros((size, size, size), dtype=np.int32)
    for x in range(size):
        for y in range(size):
            for z in range(size):
                index = 0
                for i in range(bits):
                    index |= ((x >> i) & 1) << (3 * i + 2)
                    index |= ((y >> i) & 1) << (3 * i + 0)
                    index |= ((size - 1 - z >> i) & 1) << (3 * i + 1)
                z_curve_map[x, y, z] = index
    return z_curve_map

# Precompute Xbox 360 swizzling offsets
def precompute_swizzle_offsets(size):
    swizzle_map = np.zeros((size, size, size), dtype=np.int32)
    for z in range(size):
        for x in range(size):
            for y in range(size):
                y_inverted = size - 1 - y

                # Extract individual bits of x, y, z
                x0 = (x >> 0) & 0b1
                x1 = (x >> 1) & 0b1
                x2 = (x >> 2) & 0b1
                x3 = (x >> 3) & 0b1
                x4 = (x >> 4) & 0b1
                y0 = (y_inverted >> 0) & 0b1
                y1 = (y_inverted >> 1) & 0b1
                y2 = (y_inverted >> 2) & 0b1
                y3 = (y_inverted >> 3) & 0b1
                y4 = (y_inverted >> 4) & 0b1
                z0 = (z >> 0) & 0b1
                z1 = (z >> 1) & 0b1
                z2 = (z >> 2) & 0b1
                z3 = (z >> 3) & 0b1
                z4 = (z >> 4) & 0b1

                # Calculate the offset
                offset = (
                    (z4 << 14) |
                    (z3 << 13) |
                    (z2 << 12) |
                    (y4 << 11) |
                    (z1 << 10) |
                    ((z2 << 9) ^ (y3 << 9)) |
                    (z0 << 8) |
                    (y2 << 7) |
                    (y1 << 6) |
                    ((z2 << 5) ^ (y3 << 5) ^ (x4 << 5)) |
                    (x3 << 4) |
                    (x2 << 3) |
                    (y0 << 2) |
                    (x1 << 1) |
                    (x0 << 0)
                )

                swizzle_map[z, x, y] = offset

    return swizzle_map

def load_cube(filename):
    global platform
    
    # Read the file and extract the LUT size and data points in a single pass
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # Initialize variables
    size = None
    data_points = []
    
    # Efficiently parse the file
    for line in lines:
        line = line.split('#')[0].strip()  # Remove comments and extra spaces
        if not line:
            continue
        
        if line.startswith('LUT_3D_SIZE'):
            size = int(line.split()[1])
        elif 'DOMAIN_MIN' not in line and 'DOMAIN_MAX' not in line and 'TITLE' not in line:
            data_points.append([float(x) for x in line.split()])
    
    if size is None:
        raise ValueError("LUT size (LUT_3D_SIZE) not found in the .cube file.")
    
    # Convert data points to a NumPy array and scale to [0, 255]
    data = np.array(data_points, dtype=np.float32)
    data = np.clip(data, 0.0, 1.0)  # Ensure values are within [0.0, 1.0]
    data = (data * 255).astype(np.uint8)  # Scale and convert to uint8

    # Validate the data length
    expected_points = size ** 3
    if len(data) != expected_points:
        raise ValueError(f"Expected {expected_points} data points, but got {len(data)}.")
    
    # Resample the data if size is not 32x32x32
    if size != 32:
        new_size = 32

        # Reshape the data into a 3D grid (size, size, size, 3)
        data = data.reshape((size, size, size, 3))

        # Use advanced NumPy indexing to create the new grid for resampling
        # Create indices for the new grid (size 32) and find the corresponding indices in the original grid
        x = np.linspace(0, size-1, new_size)
        y = np.linspace(0, size-1, new_size)
        z = np.linspace(0, size-1, new_size)
        
        # Use meshgrid to generate a grid of indices in the original data size
        grid_x, grid_y, grid_z = np.meshgrid(x, y, z, indexing='ij')

        # Interpolate by indexing the data with the corresponding indices in the original LUT
        resized_data = data[grid_x.astype(int), grid_y.astype(int), grid_z.astype(int)]

        # Flatten the resized LUT back to a 1D array
        data = resized_data.reshape((new_size ** 3, 3))

    # Initialize the destination array
    color_cube = data  # Shape (num_coords, 3), each entry is [R, G, B]
    dest = np.zeros_like(color_cube)

    # Efficient flattening of mapping arrays (no need for additional copies)
    indices_c = cube_map.flatten()
    indices_z = z_curve_map.flatten()
    indices_s = swizzle_map.flatten()

    # Apply the appropriate mapping
    if platform == 'PS3':
        dest[indices_z] = color_cube[indices_c]
    elif platform == 'Xbox':
        dest[indices_s] = color_cube[indices_c]
    else:
        raise ValueError("Unsupported platform. Please choose 'PS3' or 'Xbox'.")
    
    return dest

def save_cube(filename, color_cube, size=32):
    global platform

    data = np.zeros_like(color_cube)
    indices_c = cube_map.flatten()  
    indices_z = z_curve_map.flatten()  
    indices_s = swizzle_map.flatten()  

    if platform == 'PS3':
        # Apply the Z-curve mapping for PS3 platform
        data[indices_c] = color_cube[indices_z]
    elif platform == 'Xbox':
        # Apply the Swizzle mapping for Xbox platform
        data[indices_c] = color_cube[indices_s]
    else:
        raise ValueError("Unsupported platform. Please choose 'PS3' or 'Xbox'.")
    
    # Ensure data is in [0, 1] range for .cube format
    data = data.astype(np.float32) / 255.0
    data = np.clip(data, 0.0, 1.0)

    with open(filename, 'w') as f:
        # Write the .cube file header
        title = os.path.splitext(os.path.basename(filename))[0]
        f.write(f"#Generated by Skate Colourcube Editor\n")
        f.write(f"TITLE \"{title}\"\n")
        f.write("\n")
        f.write(f"LUT_3D_SIZE {size}\n")
        f.write("\n")
        f.write("DOMAIN_MIN 0.0 0.0 0.0\n")
        f.write("DOMAIN_MAX 1.0 1.0 1.0\n")
        f.write("\n")

        # Write the LUT data points
        for rgb in data:
            f.write(f"{rgb[0]:.6f} {rgb[1]:.6f} {rgb[2]:.6f}\n")

    print(f"LUT successfully saved to {filename}")


def swap_target_lut_platform(target_lut, platform):
    global z_curve_map, swizzle_map
    dest = np.zeros_like(target_lut)
    indices_z = z_curve_map.flatten()
    indices_s = swizzle_map.flatten()
    if platform == 'PS3':
        dest[indices_z] = target_lut[indices_s]
        return dest
    elif platform == 'Xbox':
        dest[indices_s] = target_lut[indices_z]
        return dest
    else:
        raise ValueError("Unsupported platform. Please choose 'PS3' or 'Xbox'.")

# Load the color cube .rgb file
def load_rgb_cube(filename):
    global target_lut, current_lut_file_path, transformed_image, blended_image, z_curve_map
    current_lut_file_path = filename
    if os.path.splitext(filename)[1].lower() == '.cube':
        color_cube = load_cube(filename)
    else:
        with open(filename, 'rb') as f:
            color_cube = np.frombuffer(f.read(), dtype=np.uint8)
            num_coords = size ** 3
            color_cube = color_cube.reshape((num_coords, 3))  # Shape (num_coords, 3), each entry is [B, G, R]
    
    color_cube = color_cube[:, [2, 1, 0]]
    target_lut = color_cube

    if image:
        transformed_image = map_image_to_color_cube(original_image, target_lut, 32, platform, z_curve_map, swizzle_map)
        # Apply changes and show the image
        apply_transformation()

def map_image_to_color_cube(img_data, target_color_cube, size, platform, z_curve_map, swizzle_map, num_threads=4):
    # Precompute scale factor
    scale_factor = (size - 1) / 255.0

    # Convert image data to NumPy arrays (instead of CuPy)
    img_data = np.array(img_data, dtype=np.float32)

    # Ensure target_color_cube is a NumPy array
    target_color_cube = np.array(target_color_cube)

    # Select precomputed map based on platform (PS3 or Xbox)
    if platform == 'PS3':
        precomputed_indices = np.array(z_curve_map)
    elif platform == 'Xbox':
        precomputed_indices = np.array(swizzle_map)
    else:
        raise ValueError("Unsupported platform selected.")

    # Determine chunk size based on the number of threads
    height, width, _ = img_data.shape
    chunk_height = height // num_threads

    # Initialize an array to hold the results
    result = np.empty_like(img_data)

    # Function to process a chunk
    def process_chunk(start, end):
        chunk = img_data[start:end]
        coords = chunk * scale_factor
        x, y, z = coords[:, :, 0], coords[:, :, 1], coords[:, :, 2]

        x0, y0, z0 = np.floor(x).astype(int), np.floor(y).astype(int), np.floor(z).astype(int)
        x1, y1, z1 = np.clip(x0 + 1, 0, size - 1), np.clip(y0 + 1, 0, size - 1), np.clip(z0 + 1, 0, size - 1)

        xd, yd, zd = x - x0, y - y0, z - z0

        flat_indices_x0y0z0 = precomputed_indices[x0, y0, z0]
        flat_indices_x0y0z1 = precomputed_indices[x0, y0, z1]
        flat_indices_x0y1z0 = precomputed_indices[x0, y1, z0]
        flat_indices_x0y1z1 = precomputed_indices[x0, y1, z1]
        flat_indices_x1y0z0 = precomputed_indices[x1, y0, z0]
        flat_indices_x1y0z1 = precomputed_indices[x1, y0, z1]
        flat_indices_x1y1z0 = precomputed_indices[x1, y1, z0]
        flat_indices_x1y1z1 = precomputed_indices[x1, y1, z1]

        c000 = target_color_cube[flat_indices_x0y0z0]
        c001 = target_color_cube[flat_indices_x0y0z1]
        c010 = target_color_cube[flat_indices_x0y1z0]
        c011 = target_color_cube[flat_indices_x0y1z1]
        c100 = target_color_cube[flat_indices_x1y0z0]
        c101 = target_color_cube[flat_indices_x1y0z1]
        c110 = target_color_cube[flat_indices_x1y1z0]
        c111 = target_color_cube[flat_indices_x1y1z1]

        c00 = (1 - xd)[:, :, None] * c000 + xd[:, :, None] * c100
        c01 = (1 - xd)[:, :, None] * c001 + xd[:, :, None] * c101
        c10 = (1 - xd)[:, :, None] * c010 + xd[:, :, None] * c110
        c11 = (1 - xd)[:, :, None] * c011 + xd[:, :, None] * c111

        c0 = (1 - yd)[:, :, None] * c00 + yd[:, :, None] * c10
        c1 = (1 - yd)[:, :, None] * c01 + yd[:, :, None] * c11

        result[start:end] = (1 - zd)[:, :, None] * c0 + zd[:, :, None] * c1

    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for i in range(num_threads):
            start = i * chunk_height
            end = (i + 1) * chunk_height if i < num_threads - 1 else height
            futures.append(executor.submit(process_chunk, start, end))
        
        # Wait for all threads to finish
        for future in futures:
            future.result()

    # Convert result back to uint8 (simulating the final step)
    return np.clip(result.astype(np.uint8), 0, 255)

# Blend the original and transformed images based on the strength factor
def blend_images(original, transformed, strength_factor):
    if strength_factor == 0:
        blended = original
    elif strength_factor <= 100:
        # Blend proportionally when strength_factor is 0-100
        alpha = strength_factor / 100.0
        blended = (original * (1 - alpha) + transformed * alpha)
    else:
        # Apply the transformation more strongly when strength_factor > 100 (up to 200)
        alpha = min(strength_factor / 100.0, 2.0)  # Limit alpha to a maximum of 2
        blended = (original * (1 - alpha) + transformed * alpha)
    
    # Ensure the result is within the valid range [0, 255]
    return np.clip(blended, 0, 255).astype(np.uint8)

def adjust_image(image_data, contrast_factor, brightness_factor, hue_factor):
    # Apply contrast: scale pixel values relative to the mean of the image
    mean = np.mean(image_data, axis=(0, 1), keepdims=True)
    image_data = (image_data - mean) * (contrast_factor / 100 + 1) + mean

    # Apply brightness: add a constant to each pixel's RGB values
    image_data += brightness_factor

    # Apply hue adjustment
    # Convert hue factor to radians for rotation
    hue_angle = hue_factor * (np.pi / 180)
    cos_hue = np.cos(hue_angle)
    sin_hue = np.sin(hue_angle)

    # Rotation matrix for hue adjustment
    # This rotates the RGB channels in a way that preserves the overall luminance
    rotation_matrix = np.array([
        [cos_hue + (1.0 - cos_hue) / 3, (1.0 - cos_hue) / 3 - sin_hue / np.sqrt(3), (1.0 - cos_hue) / 3 + sin_hue / np.sqrt(3)],
        [(1.0 - cos_hue) / 3 + sin_hue / np.sqrt(3), cos_hue + (1.0 - cos_hue) / 3, (1.0 - cos_hue) / 3 - sin_hue / np.sqrt(3)],
        [(1.0 - cos_hue) / 3 - sin_hue / np.sqrt(3), (1.0 - cos_hue) / 3 + sin_hue / np.sqrt(3), cos_hue + (1.0 - cos_hue) / 3]
    ])

    # Apply the rotation matrix to each pixel's RGB values
    image_data = np.dot(image_data, rotation_matrix.T)

    # Clip values to ensure they are valid RGB (between 0 and 255)
    return np.clip(image_data, 0, 255).astype(np.uint8)


# Save the mapped image to a new file
def save_image():
    global transformed_image, original_image
    if image is None or transformed_image is None or transformed_image.size == 0:
        print("No image or colourcube loaded.")
        return

    save_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG Files", "*.png")])
    if not save_path:
        return

    # Get the current slider values
    contrast_factor = contrast_slider.get()
    brightness_factor = brightness_slider.get()
    hue_factor = hue_slider.get()
    strength_factor = strength_slider.get()

    # Apply adjustments and blend the original and transformed images
    if contrast_factor != 0 or brightness_factor != 0 or hue_factor != 0:
        adjusted_image = adjust_image(transformed_image, contrast_factor, brightness_factor, hue_factor)
    else:
        adjusted_image = transformed_image
    
    if strength_factor != 100:
        final_image = blend_images(original_image, adjusted_image, strength_factor)
    else:
        final_image = adjusted_image

    # Save the final blended image
    Image.fromarray(final_image).save(save_path)
    print(f"Image saved to {save_path}")

def z_curve_index(x, y, z, size):
    bits = size.bit_length() - 1
    index = 0
    for i in range(bits):
        # Interleave the bits of x, y, and z
        index |= ((x >> i) & 1) << (3 * i + 2)
        index |= ((y >> i) & 1) << (3 * i + 0)
        index |= ((size - 1 - z >> i) & 1) << (3 * i + 1)
    return index

# Function to create the original LUT (color cube)
def create_og_lut():
    global swizzle_map, z_curve_map
    data = [0] * (num_coords * 3)  # 3 bytes per pixel (RGB)

    # Choose the appropriate precomputed map based on the platform
    if platform == 'PS3':
        precomputed_map = z_curve_map
    elif platform == 'Xbox':
        precomputed_map = swizzle_map
    else:
        raise ValueError("Unsupported platform selected.")

    # Iterate over the 32x32x32 reduced color cube
    for x in range(size):
        for y in range(size):
            for z in range(size):
                # Map the reduced (x, y, z) to the original (256, 256, 256) range
                color = (int(x * scale_factor), int(y * scale_factor), int(z * scale_factor))

                # Get the index from the precomputed map
                index = precomputed_map[x, y, z]

                # Store the color in the array (RGB values)
                data[index * 3:(index + 1) * 3] = [color[2], color[1], color[0]]  # Convert RGB to BGR format for storage

    return np.array(data).reshape((num_coords, 3))  # Reshape to (num_coords, 3)

# Save the mapped rgb to a new file
def save_rgb():
    global target_lut

    # Create the original LUT
    og_lut = create_og_lut()
    name = os.path.splitext(os.path.basename(current_lut_file_path))[0]

    filename = filedialog.asksaveasfilename(
    defaultextension=".rgb", 
    filetypes=[("Skate CLUT Files", "*.rgb"), ("CLUT Files", "*.cube")],
    initialfile=name
    )

    if target_lut is None:
        print("No LUT loaded.")
        target_lut = og_lut[:, [2, 1, 0]]

    if not filename:
        return

    # Get the current slider values
    contrast_factor = contrast_slider.get()
    brightness_factor = brightness_slider.get()
    hue_factor = hue_slider.get()

    # Apply adjustments and blend the original and transformed LUTs
    target_lut = adjust_image(target_lut, contrast_factor, brightness_factor, hue_factor)  # Adjust the LUT as required

    # Adjust the LUT by applying the contrast and brightness adjustments
    adjusted_lut = target_lut[:, [2, 1, 0]]  # Convert from BGR to RGB

    # Blend the original and adjusted LUTs
    final_lut = blend_images(og_lut, adjusted_lut, strength_slider.get())

    # Reshape the final LUT to be a flat array of RGB values
    final_lut_flat = final_lut.reshape(-1, 3)  # Ensure it's a flat array where each row is [R, G, B]

    # Save the RGB data as raw bytes (no header)
    if os.path.splitext(filename)[1].lower() == '.cube':
        save_cube(filename, final_lut_flat)
    else:
        with open(filename, 'wb') as f:
            # Write the RGB values as raw bytes (one byte per color channel, R8G8B8 format)
            for color in final_lut_flat:
                # Write each color channel (R, G, B) as a byte
                f.write(bytes(color))  # `bytes(color)` converts the 3-value tuple to bytes

    print(f"Colourcube saved to {filename}")

# Apply transformation in a separate thread for higher performance
def apply_transformation(*args):
    global transformed_image, blended_image, last_loaded_lut, platform_swap
    if image is None or target_lut is None:
        return
    
    if last_loaded_lut == None:
        last_loaded_lut = current_lut_file_path

    # Map image using the LUT in a background thread
    if (
        last_loaded_lut != current_lut_file_path or 
        transformed_image is None or 
        platform_swap
    ):
        platform_swap = False
        print("Mapping image with the current LUT...")
        # Divide the image into chunks and process them in parallel
        transformed_image = map_image_to_color_cube(original_image, target_lut, 32, platform, z_curve_map, swizzle_map)

    # Get the current slider values
    contrast_factor = contrast_slider.get()
    brightness_factor = brightness_slider.get()
    hue_factor = hue_slider.get()
    strength_factor = strength_slider.get()

    # Apply adjustments and blend the original and transformed images
    if contrast_factor != 0 or brightness_factor != 0 or hue_factor != 0:
        adjusted_image = adjust_image(transformed_image, contrast_factor, brightness_factor, hue_factor)
    else:
        adjusted_image = transformed_image
    
    if strength_factor != 100:
        blended_image = blend_images(original_image, adjusted_image, strength_factor)
    else:
        blended_image = adjusted_image

    # Update the displayed image
    show_image(Image.fromarray(blended_image))

def load_image(file_path):
    global original_image, image, transformed_image, blended_image
    image = Image.open(file_path).convert("RGB")
    update_image_label(file_path)
    original_image = np.array(image)  # Keep the original image as a NumPy array for transformation
    if blended_image is None:
        show_image(image)
    transformed_image = None
    apply_transformation()
    

# GUI functions for file selection and image display
def open_image():
    global image, blended_image
    blended_image = None
    file_path = filedialog.askopenfilename(title="Select an Image", filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
    load_image_files_from_directory(os.path.dirname(file_path))
    if file_path:
        load_image(file_path)

def show_image(image):
    if image is None:
        return
    
    # Ensure image is a PIL Image
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    # Scale the image based on the current zoom level
    img_resized = image.resize(
        (int(image.width * zoom_level), int(image.height * zoom_level)),
        Image.Resampling.LANCZOS
    )

    # Convert to Tkinter format
    img_tk = ImageTk.PhotoImage(img_resized)

    # Update canvas with the resized image and applied offsets
    canvas.create_image(x_offset, y_offset, anchor='nw', image=img_tk)
    canvas.image = img_tk
    canvas.config(scrollregion=canvas.bbox('all'))

def open_lut_file():
    file_path = filedialog.askopenfilename(title="Select an .rgb or .cube file", filetypes=[("CLUT Files", "*.rgb *.cube")])
    return file_path if os.path.exists(file_path) else None

# Function to divide the image into chunks
def divide_image_into_chunks(img_data, chunk_size=100):
    height, width, _ = img_data.shape
    chunks = []
    for y in range(0, height, chunk_size):
        for x in range(0, width, chunk_size):
            chunk = img_data[y:y+chunk_size, x:x+chunk_size]
            chunks.append(chunk)
    return chunks

# Combine all the chunks into a single image
def combine_chunks(chunks, img_shape, chunk_size=100):
    height, width, _ = img_shape
    result = np.zeros(img_shape, dtype=np.uint8)
    idx = 0
    for y in range(0, height, chunk_size):
        for x in range(0, width, chunk_size):
            result[y:y+chunk_size, x:x+chunk_size] = chunks[idx]
            idx += 1
    return result

def load_colourcube():
    if image is None:
        print("No image loaded.")
        return
    target_file = open_lut_file()
    if not target_file:
        print("No target LUT selected.")
        return
    update_rgb_label(target_file)
    load_lut_files_from_directory(os.path.dirname(target_file))
    load_rgb_cube(target_file)


def swap_target_lut_platform(target_lut, platform):
    global z_curve_map, swizzle_map
    dest = np.zeros_like(target_lut)
    indices_z = z_curve_map.flatten()
    indices_s = swizzle_map.flatten()
    if platform == 'PS3':
        dest[indices_z] = target_lut[indices_s]
        return dest
    elif platform == 'Xbox':
        dest[indices_s] = target_lut[indices_z]
        return dest
    else:
        raise ValueError("Unsupported platform. Please choose 'PS3' or 'Xbox'.")


# Callback for platform switch
def on_platform_change(selected_platform):
    global platform, target_lut

    if platform != selected_platform:
        platform = selected_platform
        # Swap target lut's platform if it's loaded
        if target_lut is not None:
            target_lut = swap_target_lut_platform(target_lut, platform)
            apply_transformation()

    print(f"Platform switched to: {selected_platform}")

def reset_sliders():
    # Reset all sliders to their default values
    contrast_slider.set(0)  # Default value for contrast
    brightness_slider.set(0)  # Default value for brightness
    hue_slider.set(0)  # Default value for hue
    strength_slider.set(100)  # Default value for effect strength
    show_image(Image.fromarray(transformed_image))
    print("Sliders reset to default values.")


def update_rgb_label(name):
    rgb_label.config(text=name)

def update_image_label(name):
    image_label.config(text=name)

# Global variables
current_thread = None  # Track the current thread
active_threads = []  # Track active threads for additional monitoring
last_execution_time = 0  # Timestamp of the last function execution

def terminate_thread(thread):
    """Forcefully terminate a thread."""
    if not thread.is_alive():
        return
    thread_id = ctypes.c_long(thread.ident)
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(
        thread_id, ctypes.py_object(SystemExit)
    )
    if res == 0:
        raise ValueError("Invalid thread ID")
    elif res > 1:
        # Reset to prevent affecting other threads
        ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id, None)
        raise SystemError("PyThreadState_SetAsyncExc failed")

def debounce(func):
    """Decorator to add debounce functionality to a function."""
    def wrapper(*args, **kwargs):
        global last_execution_time
        current_time = time.time()
        if current_time - last_execution_time >= 0.2:  # Debounce delay
            last_execution_time = current_time
            return func(*args, **kwargs)
    return wrapper

def is_thread_active():
    """Check if the current thread is still running."""
    global current_thread
    return current_thread is not None and current_thread.is_alive()

@debounce
def on_left_arrow(event):
    """Handle left arrow key press to cycle to the previous .rgb or .cube file."""
    global current_lut_index, current_thread, active_threads
    if lut_files:
        # Update to the previous file
        current_lut_index = (current_lut_index - 1) % len(lut_files)
        update_rgb_label(lut_files[current_lut_index])

        # If a thread is already running, do nothing
        if is_thread_active():
            return

        # Start a new thread
        current_thread = threading.Thread(target=load_rgb_cube, args=(lut_files[current_lut_index],), daemon=True)
        active_threads.append(current_thread)  # Track active threads
        current_thread.start()

@debounce
def on_right_arrow(event):
    """Handle right arrow key press to cycle to the next .rgb or .cube file."""
    global current_lut_index, current_thread, active_threads
    if lut_files:

        # Update to the next file
        current_lut_index = (current_lut_index + 1) % len(lut_files)
        update_rgb_label(lut_files[current_lut_index])

        # If a thread is already running, do nothing
        if is_thread_active():
            return

        # Start a new thread
        current_thread = threading.Thread(target=load_rgb_cube, args=(lut_files[current_lut_index],), daemon=True)
        active_threads.append(current_thread)  # Track active threads
        current_thread.start()

@debounce
def on_up_arrow(event):
    """Handle left arrow key press to cycle to the previous .rgb or .cube file."""
    global current_image_index, current_thread, active_threads
    if image_files:
        # Update to the previous file
        current_image_index = (current_image_index - 1) % len(image_files)
        update_image_label(image_files[current_image_index])

        # If a thread is already running, do nothing
        if is_thread_active():
            return

        # Start a new thread
        current_thread = threading.Thread(target=load_image, args=(image_files[current_image_index],), daemon=True)
        active_threads.append(current_thread)  # Track active threads
        current_thread.start()

@debounce
def on_down_arrow(event):
    """Handle right arrow key press to cycle to the next .rgb or .cube file."""
    global current_image_index, current_thread, active_threads
    if image_files:
        # Update to the next file
        current_image_index = (current_image_index + 1) % len(image_files)
        update_image_label(image_files[current_image_index])

        # If a thread is already running, do nothing
        if is_thread_active():
            return

        # Start a new thread
        current_thread = threading.Thread(target=load_image, args=(image_files[current_image_index],), daemon=True)
        active_threads.append(current_thread)  # Track active threads
        current_thread.start()


# GUI components
def run_app():
    global image, original_image, target_lut, transformed_image, blended_image, z_curve_map, swizzle_map, cube_map
    global strength_slider, contrast_slider, brightness_slider, hue_slider, debounced_apply_transformation
    transformed_image = None
    blended_image = None
    image = None
    target_lut = None

    cube_map = precompute_cube_order(size=32)  # Precompute cube indices 
    z_curve_map = precompute_z_curve(size=32)  # Precompute Z-curve indices
    swizzle_map = precompute_swizzle_offsets(size=32)  # Precompute swizzle indices

    root = Tk()
    root.title("Skate Color Transformation")
    root.geometry("1280x800")

    # Configure row and column weights for resizing
    root.rowconfigure(0, weight=0)  # Top row for the label (not resizable)
    root.rowconfigure(1, weight=1)  # Canvas row (resizable)
    root.rowconfigure(2, weight=0)  # Controls row (not resizable)
    
    root.columnconfigure(0, weight=1)  # Only one column, it should fill the width

    # Create a frame to hold the label above the canvas
    label_frame = Frame(root)
    label_frame.grid(row=0, column=0, sticky="w", padx=10, pady=5)  # Add padding at the top and left

    # Create the labels for displaying the file names
    global image_label
    image_label = Label(label_frame, text="", font=("Arial", 12), fg="black")
    image_label.grid(row=0, column=0, sticky="w")

    global rgb_label
    rgb_label = Label(label_frame, text="", font=("Arial", 12), fg="blue")
    rgb_label.grid(row=0, column=1, sticky="w")  # Place label at top-left of the label_frame

    # Create canvas for image display
    global canvas
    canvas = Canvas(root, bg="gray")
    canvas.grid(row=1, column=0, sticky="nsew")  # Fill available space
    canvas.bind("<ButtonPress-1>", on_drag_start)  # When mouse is pressed, start dragging
    canvas.bind("<B1-Motion>", on_drag_motion)    # When mouse is moved with button pressed, drag the image
    canvas.bind("<MouseWheel>", on_mouse_wheel)   # Zoom in/out with mouse wheel

    # Bind the left and right arrow keys to the functions
    root.bind("<Left>", on_left_arrow)  # Bind left arrow key to the on_left_arrow function
    root.bind("<Right>", on_right_arrow)  # Bind right arrow key to the on_right_arrow function
    root.bind("<Up>", on_up_arrow)  # Bind left arrow key to the on_left_arrow function
    root.bind("<Down>", on_down_arrow)  # Bind right arrow key to the on_right_arrow function


    # Create a frame for controls (buttons and sliders)
    controls_frame = Frame(root)
    controls_frame.grid(row=2, column=0, sticky="ew")  # Place at the bottom

    # Configure controls frame to expand horizontally
    root.columnconfigure(0, weight=1)

    # Add buttons and sliders to the controls frame
    Button(controls_frame, text="Open Image", command=open_image).pack(side='left', padx=5, pady=5)
    Button(controls_frame, text="Load CLUT *.rgb or *.cube", command=load_colourcube).pack(side='left', padx=5, pady=5)
    Button(controls_frame, text="Save Image", command=save_image).pack(side='left', padx=5, pady=5)

    Button(controls_frame, text="Save CLUT *.rgb or *.cube", command=save_rgb).pack(side='right', padx=5, pady=5)
    Button(controls_frame, text="↺", command=reset_sliders).pack(side='right', padx=5, pady=5)

    # Add platform switch dropdown
    platform_var = StringVar(value="PS3")  # Default platform is PS3
    platform_menu = OptionMenu(controls_frame, platform_var, "PS3", "Xbox", command=on_platform_change)
    platform_menu.pack(side='left', padx=5, pady=5)

    # Add sliders to control Effect Strength, Contrast, Brightness
    contrast_slider = Scale(controls_frame, from_=-100, to=100, orient=HORIZONTAL, label="Contrast", length=170)
    contrast_slider.set(0)
    contrast_slider.pack(side='right', padx=5, pady=5)

    brightness_slider = Scale(controls_frame, from_=-100, to=100, orient=HORIZONTAL, label="Brightness", length=170)
    brightness_slider.set(0)
    brightness_slider.pack(side='right', padx=5, pady=5)

    hue_slider = Scale(controls_frame, from_=-100, to=100, orient=HORIZONTAL, label="Hue", length=170)
    hue_slider.set(0)
    hue_slider.pack(side='right', padx=5, pady=5)

    strength_slider = Scale(controls_frame, from_=0, to=200, orient=HORIZONTAL, label="Effect Strength", length=170)
    strength_slider.set(100)  # Default value to 100 (normal effect)
    strength_slider.pack(side='right', padx=5, pady=5)

    # Initialize the debounced transformation function
    debounced_apply_transformation = Debouncer(apply_transformation, delay=0.2)  # 200ms debounce

    # Bind sliders to update image with debounce
    contrast_slider.bind("<Motion>", apply_transformation_debounced)
    brightness_slider.bind("<Motion>", apply_transformation_debounced)
    hue_slider.bind("<Motion>", apply_transformation_debounced)
    strength_slider.bind("<Motion>", apply_transformation_debounced)

    # Start the main loop
    root.mainloop()

if __name__ == "__main__":
    run_app()
