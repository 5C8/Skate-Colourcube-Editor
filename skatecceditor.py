import numpy as np
from tkinter import Tk, filedialog, Button, Canvas, Scale, OptionMenu, StringVar, HORIZONTAL, Frame
from PIL import Image, ImageTk
import os
import concurrent.futures
from threading import Timer

# Debounce implementation
class Debouncer:
    def __init__(self, func, delay):
        self.func = func
        self.delay = delay
        self.timer = None

    def call(self, *args, **kwargs):
        if self.timer:
            self.timer.cancel()  # Cancel any pending call
        self.timer = Timer(self.delay, self.func, args, kwargs)
        self.timer.start()

# Global debounce instances
debounced_apply_transformation = None

def apply_transformation_debounced(*args):
    global debounced_apply_transformation
    if debounced_apply_transformation:
        debounced_apply_transformation.call()

last_loaded_lut = None
current_lut_file_path = None

platform = 'PS3'
platform_swap = False
zoom_level = 1.0  # Start with a default zoom level
x_offset = 0  # Initial horizontal offset
y_offset = 0  # Initial vertical offset

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

# Load the color cube .rgb file
def load_rgb_cube(filename, size):
    num_coords = size ** 3
    with open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)
    
    color_cube = data.reshape((num_coords, 3))  # Shape (num_coords, 3), each entry is [B, G, R]
    color_cube = color_cube[:, [2, 1, 0]]
    return color_cube

# Map the image colors using the color cube
def map_image_to_color_cube(img_data, target_color_cube, size):
    global platform, z_curve_map, swizzle_map
    scale_factor = (size - 1) / 255.0  # Scale RGB to LUT range (0 to size-1)

    # Map RGB values to floating-point grid coordinates
    x = img_data[:, :, 0] * scale_factor
    y = img_data[:, :, 1] * scale_factor
    z = img_data[:, :, 2] * scale_factor

    # Compute floor and ceiling indices
    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    z0 = np.floor(z).astype(int)
    x1 = np.clip(x0 + 1, 0, size - 1)
    y1 = np.clip(y0 + 1, 0, size - 1)
    z1 = np.clip(z0 + 1, 0, size - 1)

    # Interpolation weights
    xd = x - x0
    yd = y - y0
    zd = z - z0

    if platform == 'PS3':
        precomputed_indices = z_curve_map
    elif platform == 'Xbox':
        precomputed_indices = swizzle_map
    else:
        raise ValueError("Unsupported platform selected.")

    # Fetch colors using precomputed indices
    c000 = target_color_cube[precomputed_indices[x0, y0, z0]]
    c001 = target_color_cube[precomputed_indices[x0, y0, z1]]
    c010 = target_color_cube[precomputed_indices[x0, y1, z0]]
    c011 = target_color_cube[precomputed_indices[x0, y1, z1]]
    c100 = target_color_cube[precomputed_indices[x1, y0, z0]]
    c101 = target_color_cube[precomputed_indices[x1, y0, z1]]
    c110 = target_color_cube[precomputed_indices[x1, y1, z0]]
    c111 = target_color_cube[precomputed_indices[x1, y1, z1]]

    # Perform trilinear interpolation
    c00 = c000 * (1 - xd)[:, :, None] + c100 * xd[:, :, None]
    c01 = c001 * (1 - xd)[:, :, None] + c101 * xd[:, :, None]
    c10 = c010 * (1 - xd)[:, :, None] + c110 * xd[:, :, None]
    c11 = c011 * (1 - xd)[:, :, None] + c111 * xd[:, :, None]

    c0 = c00 * (1 - yd)[:, :, None] + c10 * yd[:, :, None]
    c1 = c01 * (1 - yd)[:, :, None] + c11 * yd[:, :, None]

    c = c0 * (1 - zd)[:, :, None] + c1 * zd[:, :, None]

    return c.astype(np.uint8)

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

    # Apply adjustments and blend the original and transformed images
    adjusted_image = adjust_image(transformed_image, contrast_factor, brightness_factor, hue_factor)
    final_image = blend_images(original_image, adjusted_image, strength_slider.get())

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

# Size of the reduced 32x32x32 color cube
size = 32
scale_factor = 255 / (size - 1)
num_coords = size ** 3

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

    save_path = filedialog.asksaveasfilename(defaultextension=".rgb", filetypes=[("RGB Files", "*.rgb")])

    if target_lut is None:
        print("No LUT loaded.")
        target_lut = og_lut[:, [2, 1, 0]]

    if not save_path:
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
    with open(save_path, 'wb') as f:
        # Write the RGB values as raw bytes (one byte per color channel, R8G8B8 format)
        for color in final_lut_flat:
            # Write each color channel (R, G, B) as a byte
            f.write(bytes(color))  # `bytes(color)` converts the 3-value tuple to bytes

    print(f"Colourcube saved to {save_path}")

# Apply transformation in a separate thread for higher performance
def apply_transformation(*args):
    global current_transformed_image, transformed_image, blended_image, last_loaded_lut, platform_swap
    if image is None or target_lut is None:
        return
    
    if last_loaded_lut == None:
        last_loaded_lut = current_lut_file_path

    # Map image using the LUT in a background thread
    if last_loaded_lut != current_lut_file_path or transformed_image is None or platform_swap:
        platform_swap = False
        print("Mapping image with the current LUT...")
        with ThreadPoolExecutor() as executor:
            future = executor.submit(map_image_to_color_cube, original_image, target_lut, 32)
            transformed_image = future.result()

    # Get the current slider values
    contrast_factor = contrast_slider.get()
    brightness_factor = brightness_slider.get()
    hue_factor = hue_slider.get()

    # Apply adjustments and blend the images based on the current settings
    adjusted_image = adjust_image(transformed_image, contrast_factor, brightness_factor, hue_factor)
    blended_image = blend_images(original_image, adjusted_image, strength_slider.get())

    # Update the displayed image
    show_image(Image.fromarray(blended_image))

# GUI functions for file selection and image display
def open_image():
    global image, original_image, blended_image
    blended_image = None
    file_path = filedialog.askopenfilename(title="Select an Image", filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
    if file_path:
        image = Image.open(file_path).convert("RGB")
        original_image = np.array(image)  # Keep the original image as a NumPy array for transformation
        show_image(image)
        return image
    return None

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
    file_path = filedialog.askopenfilename(title="Select an .rgb File", filetypes=[("RGB Files", "*.rgb")])
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
    global target_lut, transformed_image, blended_image, z_curve_map
    if image is None:
        print("No image loaded.")
        return
    target_file = open_lut_file()
    if not target_file:
        print("No target LUT selected.")
        return
    target_lut = load_rgb_cube(target_file, size=32)  # Adjust size as needed
    if image:
        # Divide the image into chunks and process them in parallel
        chunks = divide_image_into_chunks(original_image, chunk_size=100)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_results = [executor.submit(map_image_to_color_cube, chunk, target_lut, 32) for chunk in chunks]
            results = [future.result() for future in future_results]
        transformed_image = combine_chunks(results, original_image.shape, chunk_size=100)

        # Apply changes and show the image
        apply_transformation()


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

# GUI components
def run_app():
    global image, original_image, target_lut, transformed_image, blended_image, z_curve_map, swizzle_map
    global strength_slider, contrast_slider, brightness_slider, hue_slider, debounced_apply_transformation
    transformed_image = None
    blended_image = None
    image = None
    target_lut = None

    z_curve_map = precompute_z_curve(size=32)  # Precompute Z-curve indices
    swizzle_map = precompute_swizzle_offsets(size=32)  # Precompute swizzle indices

    root = Tk()
    root.title("Skate Color Transformation")
    root.geometry("1280x800")

    # Configure row and column weights for resizing
    root.rowconfigure(0, weight=1)  # Canvas row
    root.columnconfigure(0, weight=1)  # Entire window's single column

    # Create canvas for image display
    global canvas
    canvas = Canvas(root, bg="gray")
    canvas.grid(row=0, column=0, sticky="nsew")  # Fill available space
    canvas.bind("<ButtonPress-1>", on_drag_start)  # When mouse is pressed, start dragging
    canvas.bind("<B1-Motion>", on_drag_motion)    # When mouse is moved with button pressed, drag the image
    canvas.bind("<MouseWheel>", on_mouse_wheel)   # Zoom in/out with mouse wheel

    # Create a frame for controls (buttons and sliders)
    controls_frame = Frame(root)
    controls_frame.grid(row=1, column=0, sticky="ew")  # Place at the bottom

    # Configure controls frame to expand horizontally
    root.rowconfigure(1, weight=0)
    root.columnconfigure(0, weight=1)

    # Add buttons and sliders to the controls frame
    Button(controls_frame, text="Open Image", command=open_image).pack(side='left', padx=5, pady=5)
    Button(controls_frame, text="Load Skate cc *.rgb", command=load_colourcube).pack(side='left', padx=5, pady=5)
    Button(controls_frame, text="Save Image", command=save_image).pack(side='left', padx=5, pady=5)

    Button(controls_frame, text="Save Skate cc *.rgb", command=save_rgb).pack(side='right', padx=5, pady=5)
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