import numpy as np
from tkinter import Tk, filedialog, Button, Canvas, Scale, OptionMenu, StringVar, HORIZONTAL, Frame, Label
from PIL import Image, ImageTk
import os
from concurrent.futures import ThreadPoolExecutor
import hashlib
import threading
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

current_image_tk = None
last_loaded_lut = None
current_lut_file_path = None
lut_files = []
image_files = []
current_lut_index = -1
current_image_index = -1

platform = 'PS3'
lut_reload = False
zoom_level = 1.0  # Start with a default zoom level
x_offset = 0  # Initial horizontal offset
y_offset = 0  # Initial vertical offset

# Size of the reduced 32x32x32 color cube
size = 32
scale_factor = 255 / (size - 1)
num_coords = size ** 3

def load_lut_files_from_directory(directory):
    global lut_files, current_lut_index
    lut_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.lower().endswith(('.rgb', '.cube'))]
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
    image_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
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

def parse_cube_file(file_path):
    data_points = []
    size = None
    parsing_data = False  # Flag to track when to start parsing data points
    
    with open(file_path, 'r') as file:
        for line in file:   
            if not parsing_data:
                line = line.split('#', 1)[0].strip()  # Remove comments and leading/trailing whitespaces
                if not line:  # Skip empty lines
                    continue
                # Process LUT_3D_SIZE to extract the size
                if line.startswith('LUT_3D_SIZE'):
                    size = int(line.split()[1])
                    continue  # Skip processing this line for data
                
                # Skip lines that contain 'DOMAIN_MIN', 'DOMAIN_MAX', 'TITLE', etc.
                if 'DOMAIN_MIN' in line or 'DOMAIN_MAX' in line or 'TITLE' in line:
                    continue
                
                # The first valid data line is encountered, start parsing data points
                parsing_data = True
                data_points.append(tuple(map(float, line.split())))
            else:
                data_points.append(tuple(map(float, line.split())))
    
    return size, data_points


def load_cube(file_path):
    global platform
    
    # Initialize variables
    size = None
    data_points = []
    
    size, data_points = parse_cube_file(file_path)

    if size is None:
        raise ValueError("LUT size (LUT_3D_SIZE) not found in the .cube file.")
    
    # Convert data points to a NumPy array and scale to [0, 255]
    data = (np.array(data_points, dtype=np.float32) * 255).astype(np.uint8)

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
    
    dest = dest[:, [0, 1, 2]]
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
    global target_lut, current_lut_file_path, blended_image, z_curve_map
    current_lut_file_path = filename
    if os.path.splitext(filename)[1].lower() == '.cube':
        color_cube = load_cube(filename)
    else:
        with open(filename, 'rb') as f:
            color_cube = np.frombuffer(f.read(), dtype=np.uint8)
            num_coords = size ** 3
            color_cube = color_cube.reshape((num_coords, 3))  # Shape (num_coords, 3), each entry is [B, G, R]
    
    target_lut = color_cube
    apply_transformation()

# Load neutral color cube
def load_neutral_cube():
    global target_lut, current_lut_file_path, blended_image, z_curve_map, lut_reload, current_lut_file_path
    og_lut = create_og_lut()
    update_rgb_label('neutral')
    current_lut_file_path = 'neutral'
    target_lut = og_lut[:, [2, 1, 0]]
    lut_reload = True
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
        z, y, x = coords[:, :, 0], coords[:, :, 1], coords[:, :, 2]

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

# Function to blend a chunk of the image
def blend_chunk(original_chunk, transformed_chunk, alpha):
    return (original_chunk * (1 - alpha) + transformed_chunk * alpha)

# Main function to blend images with multithreading
def blend_images(original, transformed, strength_factor, num_workers=12):
    height = original.shape[0]
    chunk_size = height // num_workers
    remainder = height % num_workers
    
    # If strength_factor is 0, return the original image
    if strength_factor == 0:
        return original
    
    # Calculate the alpha based on the strength_factor
    if strength_factor <= 100:
        alpha = strength_factor / 100.0
    else:
        alpha = min(strength_factor / 100.0, 2.0)  # Limit alpha to a maximum of 2

    # Divide the images into chunks
    chunks_original = [original[i * chunk_size:(i + 1) * chunk_size] for i in range(num_workers)]
    chunks_transformed = [transformed[i * chunk_size:(i + 1) * chunk_size] for i in range(num_workers)]

    # Handle the remainder for the last chunk
    if remainder > 0:
        chunks_original[-1] = original[-remainder:]
        chunks_transformed[-1] = transformed[-remainder:]

    # Use ThreadPoolExecutor to process chunks concurrently
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        blended_chunks = list(executor.map(blend_chunk, chunks_original, chunks_transformed, [alpha] * num_workers))

    # Reconstruct the blended image by combining all the chunks
    blended = np.vstack(blended_chunks)

    # Ensure the result is within the valid range [0, 255]
    return np.clip(blended, 0, 255).astype(np.uint8)

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

def adjust_saturation_chunk(chunk, saturation_factor):
    if saturation_factor == 0:
        return chunk

    # Normalize the image data to [0, 1]
    chunk = chunk / 255.0
    R, G, B = chunk[..., 0], chunk[..., 1], chunk[..., 2]

    # Calculate max, min, delta, and lightness
    max_val = np.max(chunk, axis=-1)
    min_val = np.min(chunk, axis=-1)
    delta = max_val - min_val
    L = (max_val + min_val) / 2

    # Adjust saturation using a vectorized approach
    epsilon = 1e-10
    S = np.where(L < 0.5, delta / (max_val + min_val + epsilon), delta / (2 - max_val - min_val + epsilon))
    S = np.nan_to_num(S)  # Avoid NaNs from division by zero
    S = np.clip(S * (saturation_factor / 100 + 1), 0, 1)

    # Precompute constants
    one_sixth = 1 / 6
    two_thirds = 2 / 3

    # Calculate Q and P for hue adjustment
    Q = np.where(L < 0.5, L * (1 + S), L + S - L * S)
    P = 2 * L - Q

    # Vectorized hue calculations
    H = np.zeros_like(R)
    idx = delta > 0
    delta = np.where(delta == 0, epsilon, delta)  # Avoid division by zero in hue calculation
    H[idx & (max_val == R)] = (G - B)[idx & (max_val == R)] / delta[idx & (max_val == R)]
    H[idx & (max_val == G)] = 2 + (B - R)[idx & (max_val == G)] / delta[idx & (max_val == G)]
    H[idx & (max_val == B)] = 4 + (R - G)[idx & (max_val == B)] / delta[idx & (max_val == B)]
    H /= 6
    H = np.where(H < 0, H + 1, H)

    # Vectorized _hue_to_rgb function
    def _hue_to_rgb(p, q, t):
        t = np.where(t < 0, t + 1, t)
        t = np.where(t > 1, t - 1, t)
        return np.where(t < one_sixth, p + (q - p) * 6 * t,
               np.where(t < 0.5, q, 
               np.where(t < two_thirds, p + (q - p) * (two_thirds - t) * 6, p)))

    R = _hue_to_rgb(P, Q, H + one_sixth)
    G = _hue_to_rgb(P, Q, H)
    B = _hue_to_rgb(P, Q, H - one_sixth)

    # Reconstruct and scale the image back to [0, 255]
    chunk = np.stack([R, G, B], axis=-1) * 255
    return np.clip(chunk, 0, 255).astype(np.uint8)

def adjust_saturation(image_data, saturation_factor, num_workers=12):
    height = image_data.shape[0]
    
    # Calculate the chunk size and the remainder
    chunk_size = height // num_workers
    remainder = height % num_workers

    # Slice image_data into chunks, with the last chunk taking the remainder
    chunks = [image_data[i * chunk_size:(i + 1) * chunk_size] for i in range(num_workers)]
    
    # Handle the remainder by adding it to the last chunk
    if remainder > 0:
        chunks[-1] = np.vstack([chunks[-1], image_data[-remainder:]])

    # Process the chunks using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        processed_chunks = list(executor.map(adjust_saturation_chunk, chunks, [saturation_factor] * num_workers))

    return np.vstack(processed_chunks)

saturated_image = None
desaturated_image = None
previous_image_hash = None

def hash_image(image_data):
    """Generate a hash for the given image data."""
    return hashlib.sha256(image_data.tobytes()).hexdigest()

def adjust_image(image_data, saturation_factor, contrast_factor, brightness_factor, hue_factor):
    global saturated_image, desaturated_image, previous_image_hash
    
    current_image_hash = hash_image(image_data)
    if previous_image_hash != current_image_hash:
        # Image data has changed, recalculate saturated and desaturated images
        saturated_image = adjust_saturation(image_data, 200)
        desaturated_image = adjust_saturation(image_data, -100)
        previous_image_hash = current_image_hash

    if saturation_factor != 0:
        if saturation_factor > 0:
            image_data = blend_images(image_data, saturated_image, saturation_factor)
        else:
            image_data = blend_images(image_data, desaturated_image, -saturation_factor)

    # Apply contrast adjustment
    mean = np.mean(image_data, axis=(0, 1), keepdims=True)
    image_data = (image_data - mean) * (contrast_factor / 100 + 1) + mean

    # Apply brightness adjustment
    image_data += brightness_factor

    # Apply hue adjustment
    hue_angle = hue_factor * (np.pi / 180)
    cos_hue = np.cos(hue_angle)
    sin_hue = np.sin(hue_angle)
    rotation_matrix = np.array([
        [cos_hue + (1.0 - cos_hue) / 3, (1.0 - cos_hue) / 3 - sin_hue / np.sqrt(3), (1.0 - cos_hue) / 3 + sin_hue / np.sqrt(3)],
        [(1.0 - cos_hue) / 3 + sin_hue / np.sqrt(3), cos_hue + (1.0 - cos_hue) / 3, (1.0 - cos_hue) / 3 - sin_hue / np.sqrt(3)],
        [(1.0 - cos_hue) / 3 - sin_hue / np.sqrt(3), (1.0 - cos_hue) / 3 + sin_hue / np.sqrt(3), cos_hue + (1.0 - cos_hue) / 3]
    ])
    image_data = np.dot(image_data, rotation_matrix.T)
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
    saturation_factor = saturation_slider.get()
    contrast_factor = contrast_slider.get()
    brightness_factor = brightness_slider.get()
    hue_factor = hue_slider.get()
    strength_factor = strength_slider.get()

    # Apply adjustments and blend the original and transformed images
    if saturation_factor != 0 or contrast_factor != 0 or brightness_factor != 0 or hue_factor != 0:
        adjusted_image = adjust_image(transformed_image, saturation_factor, contrast_factor, brightness_factor, hue_factor)
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
                data[index * 3:(index + 1) * 3] = [color[0], color[1], color[2]]  # Convert RGB to BGR format for storage

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
        target_lut = og_lut

    if not filename:
        return

    # Get the current slider values
    saturation_factor = saturation_slider.get()
    contrast_factor = contrast_slider.get()
    brightness_factor = brightness_slider.get()
    hue_factor = hue_slider.get()

    # Apply adjustments and blend the original and transformed LUTs
    target_lut = adjust_image(target_lut, saturation_factor, contrast_factor, brightness_factor, hue_factor)  # Adjust the LUT as required

    # Blend the original and adjusted LUTs
    final_lut = blend_images(og_lut, target_lut, strength_slider.get())

    # Reshape the final LUT to be a flat array of RGB values
    final_lut_flat = final_lut[:, [0, 1, 2]].reshape(-1, 3)  # Ensure it's a flat array where each row is [R, G, B]

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
    global transformed_image, blended_image, last_loaded_lut, lut_reload
    if image is None or target_lut is None:
        return
    
    if last_loaded_lut == None:
        last_loaded_lut = current_lut_file_path

    # Map image using the LUT in a background thread
    if (
        last_loaded_lut != current_lut_file_path or 
        transformed_image is None or 
        lut_reload
    ):
        lut_reload = False
        # Divide the image into chunks and process them in parallel
        if current_lut_file_path == 'neutral':
            transformed_image = original_image
        else:
            print("Mapping image with the current LUT...")
            transformed_image = map_image_to_color_cube(original_image, target_lut, 32, platform, z_curve_map, swizzle_map)
        last_loaded_lut = current_lut_file_path

    # Get the current slider values
    saturation_factor = saturation_slider.get()
    contrast_factor = contrast_slider.get()
    brightness_factor = brightness_slider.get()
    hue_factor = hue_slider.get()
    strength_factor = strength_slider.get()

    # Apply adjustments and blend the original and transformed images
    if saturation_factor != 0 or contrast_factor != 0 or brightness_factor != 0 or hue_factor != 0:
        adjusted_image = adjust_image(transformed_image, saturation_factor, contrast_factor, brightness_factor, hue_factor)
    else:
        adjusted_image = transformed_image
    
    if strength_factor != 100:
        blended_image = blend_images(original_image, adjusted_image, strength_factor)
    else:
        blended_image = adjusted_image

    # Update the displayed image
    show_image(Image.fromarray(blended_image))

def load_image(file_path):
    global original_image, image, transformed_image, blended_image, zoom_level
    image = Image.open(file_path).convert("RGB")
    update_image_label(file_path)
    original_image = np.array(image)  # Keep the original image as a NumPy array for transformation

    canvas_width = canvas.winfo_width()
    canvas_height = canvas.winfo_height()
    img_width, img_height = image.size
    scale_x = canvas_width / img_width
    scale_y = canvas_height / img_height

    # Choose the smaller scale factor to fit the image inside the canvas
    zoom_level = min(scale_x, scale_y)

    if blended_image is None:
        show_image(image)
    transformed_image = None
    if target_lut is None:
        load_neutral_cube()
        return
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
    global current_image_tk
    
    if image is None:
        return

    # Ensure image is a PIL Image
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    # Scale the image based on the current zoom level only if necessary
    img_resized = image.resize(
        (int(image.width * zoom_level), int(image.height * zoom_level)),
        Image.Resampling.LANCZOS
    )

    # Convert to Tkinter format
    img_tk = ImageTk.PhotoImage(img_resized)

    # Reuse the previous image if it has not changed
    if current_image_tk != img_tk:
        # Update the canvas with the resized image and applied offsets
        canvas.create_image(x_offset, y_offset, anchor='center', image=img_tk)
        canvas.image = img_tk
        current_image_tk = img_tk  # Keep track of the current image

    # Update the scroll region
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
    saturation_slider.set(0)
    contrast_slider.set(0)  # Default value for contrast
    brightness_slider.set(0)  # Default value for brightness
    hue_slider.set(0)  # Default value for hue
    strength_slider.set(100)  # Default value for effect strength
    #show_image(Image.fromarray(transformed_image))
    print("Sliders reset to default values.")


def update_rgb_label(name):
    rgb_label.config(text=name)

def update_image_label(name):
    image_label.config(text=name)

def on_mouse_wheel_slider(event, slider):
    """Handle mouse scroll to adjust slider value."""
    if event.delta > 0:
        slider.set(slider.get() + 1)  # Scroll up, increase value
    elif event.delta < 0:
        slider.set(slider.get() - 1)

# Global variables
current_thread = None  # Track the current thread
active_threads = []  # Track active threads for additional monitoring
last_execution_time = 0  # Timestamp of the last function execution

def debounce(func):
    def wrapper(*args, **kwargs):
        global last_execution_time
        current_time = time.time()
        if current_time - last_execution_time >= 0.2:  # Debounce delay
            last_execution_time = current_time
            return func(*args, **kwargs)
    return wrapper

def is_thread_active():
    global current_thread
    return current_thread is not None and current_thread.is_alive()

@debounce
def on_left_arrow(event):
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
    global strength_slider, saturation_slider, contrast_slider, brightness_slider, hue_slider, debounced_apply_transformation
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
    image_l1 = Label(label_frame, text="[", font=("Arial", 12), fg="black")
    image_l1.grid(row=0, column=0, sticky="w")
    global image_label
    image_label = Label(label_frame, text="-", font=("Arial", 12), fg="green")
    image_label.grid(row=0, column=1, sticky="w")
    image_l2 = Label(label_frame, text="]", font=("Arial", 12), fg="black")
    image_l2.grid(row=0, column=2, sticky="w")

    rgb_l1 = Label(label_frame, text="[", font=("Arial", 12), fg="black")
    rgb_l1.grid(row=0, column=3, sticky="w")
    global rgb_label
    rgb_label = Label(label_frame, text="-", font=("Arial", 12), fg="blue")
    rgb_label.grid(row=0, column=4, sticky="w")  # Place label at top-left of the label_frame
    rgb_l2 = Label(label_frame, text="]", font=("Arial", 12), fg="black")
    rgb_l2.grid(row=0, column=5, sticky="w")

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
    Button(controls_frame, text="Load LUT", command=load_colourcube).pack(side='left', padx=5, pady=5)
    Button(controls_frame, text="Neutral LUT", command=load_neutral_cube).pack(side='left', padx=5, pady=5)
    Button(controls_frame, text="Save Image", command=save_image).pack(side='left', padx=5, pady=5)

    Button(controls_frame, text="Save LUT", command=save_rgb).pack(side='right', padx=5, pady=5)
    Button(controls_frame, text="↺", command=reset_sliders).pack(side='right', padx=5, pady=5)

    # Add platform switch dropdown
    platform_var = StringVar(value="PS3")  # Default platform is PS3
    platform_menu = OptionMenu(controls_frame, platform_var, "PS3", "Xbox", command=on_platform_change)
    platform_menu.pack(side='left', padx=5, pady=5)

    # Add sliders to control Effect Strength, Contrast, Brightness
    saturation_slider = Scale(controls_frame, from_=-100, to=100, orient=HORIZONTAL, label="Saturation", length=170)
    saturation_slider.set(0)
    saturation_slider.pack(side='right', padx=5, pady=5)
    saturation_slider.bind("<MouseWheel>", lambda event, slider=saturation_slider: on_mouse_wheel_slider(event, slider))

    contrast_slider = Scale(controls_frame, from_=-100, to=100, orient=HORIZONTAL, label="Contrast", length=170)
    contrast_slider.set(0)
    contrast_slider.pack(side='right', padx=5, pady=5)
    contrast_slider.bind("<MouseWheel>", lambda event, slider=contrast_slider: on_mouse_wheel_slider(event, slider))

    brightness_slider = Scale(controls_frame, from_=-100, to=100, orient=HORIZONTAL, label="Brightness", length=170)
    brightness_slider.set(0)
    brightness_slider.pack(side='right', padx=5, pady=5)
    brightness_slider.bind("<MouseWheel>", lambda event, slider=brightness_slider: on_mouse_wheel_slider(event, slider))

    hue_slider = Scale(controls_frame, from_=-100, to=100, orient=HORIZONTAL, label="Hue", length=170)
    hue_slider.set(0)
    hue_slider.pack(side='right', padx=5, pady=5)
    hue_slider.bind("<MouseWheel>", lambda event, slider=hue_slider: on_mouse_wheel_slider(event, slider))

    strength_slider = Scale(controls_frame, from_=0, to=200, orient=HORIZONTAL, label="Effect Strength", length=170)
    strength_slider.set(100)  # Default value to 100 (normal effect)
    strength_slider.pack(side='right', padx=5, pady=5)
    strength_slider.bind("<MouseWheel>", lambda event, slider=strength_slider: on_mouse_wheel_slider(event, slider))

    # Initialize the debounced transformation function
    debounced_apply_transformation = Debouncer(apply_transformation, delay=0.2)  # 200ms debounce

    # Bind sliders to update image with debounce
    saturation_slider.bind("<Motion>", apply_transformation_debounced)
    contrast_slider.bind("<Motion>", apply_transformation_debounced)
    brightness_slider.bind("<Motion>", apply_transformation_debounced)
    hue_slider.bind("<Motion>", apply_transformation_debounced)
    strength_slider.bind("<Motion>", apply_transformation_debounced)

    # Start the main loop
    root.mainloop()

if __name__ == "__main__":
    run_app()
