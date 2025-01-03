# Helper function to perform xbox360 swizzling
def calc_offset32(z, x, y, size):
    y = size - 1 - y

    # Extract individual bits of x, y, z
    x0 = (x >> 0) & 0b1
    x1 = (x >> 1) & 0b1
    x2 = (x >> 2) & 0b1
    x3 = (x >> 3) & 0b1
    x4 = (x >> 4) & 0b1
    y0 = (y >> 0) & 0b1
    y1 = (y >> 1) & 0b1
    y2 = (y >> 2) & 0b1
    y3 = (y >> 3) & 0b1
    y4 = (y >> 4) & 0b1
    z0 = (z >> 0) & 0b1
    z1 = (z >> 1) & 0b1
    z2 = (z >> 2) & 0b1
    z3 = (z >> 3) & 0b1
    z4 = (z >> 4) & 0b1

    # Calculate the offset
    o = (
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

    return o

# Size of the reduced 32x32x32 color cube
size = 32
scale_factor = 255 / (size - 1)  # Scale to 0-255 range
num_coords = size ** 3  # Total number of pixels
data = [0] * (num_coords * 3)  # 3 bytes per pixel (RGB)

# Iterate over the 32x32x32 reduced color cube
for x in range(size):
    for y in range(size):
        for z in range(size):
            # Map the reduced (x, y, z) to the original (256, 256, 256) range
            color = (int(x * scale_factor), int(y * scale_factor), int(z * scale_factor))
            
            # Get the index
            index = calc_offset32(x, y, z, size)

            # Store the color in the array (RGB values)
            data[index * 3:(index + 1) * 3] = [color[2], color[1], color[0]]


# Save as a headerless R8G8B8 .rgb file
with open('cc_neutral.rgb', 'wb') as f:
    f.write(bytearray(data))
