# Helper function to perform bit-interleaving for Z-curve mapping
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
data = [0] * (num_coords * 3)  # 3 bytes per pixel (RGB)

# Iterate over the 32x32x32 reduced color cube
for x in range(size):
    for y in range(size):
        for z in range(size):
            # Map the reduced (x, y, z) to the original (256, 256, 256) range
            color = (int(x * scale_factor), int(y * scale_factor), int(z * scale_factor))
            
            # Get the index
            index = z_curve_index(x, y, z, size)

            # Store the color in the array (RGB values)
            data[index * 3:(index + 1) * 3] = [color[2], color[1], color[0]]

# Save as a headerless R8G8B8 .rgb file
with open('cc_neutral.rgb', 'wb') as f:
    f.write(bytearray(data))
