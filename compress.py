"""Image Compressor

This script converts given image files into grayscale and compresses them using
Fourier transforms. Image dimensions are cropped to a multiple of 32.

This script requires that `numpy` and 'pillow' be installed within the Python
environment.

This file can also be imported as a module and contains the following
functions:
    * compress - compresses a grayscale image given as a numpy array
    * main - the main function of the script
"""

import argparse
import numpy as np
from PIL import Image


COEFFICIENT_TOLERANCE_VALUE = 0.0605


def compress(image, tol):
    block_rows = np.shape(image)[0]//32
    block_cols = np.shape(image)[1]//32
    result = np.zeros(shape=(block_rows*32,block_cols*32))
    
    old_nonzero = 0
    new_nonzero = 0
    
    for i in range(block_rows):
        for j in range(block_cols):
            # Acquire the current 32x32 sub-block for processing
            block = image[i*32:(i+1)*32, j*32:(j+1)*32]
            
            # Find the 2D Fourier coefficients
            F = np.fft.fft2(block)
            
            # Count the number of nonzero coefficients in current block
            old_nonzero += np.count_nonzero(abs(F))
            
            # Remove DC and keep it separate from calculations
            F_0 = F[0,0]
            F[0,0] = 0
            
            # Find F_max and use it to calculate cutoff threshold
            thresh = np.amax(abs(F)) * tol

            # Zero all values below threshold
            F = np.where(abs(F) > thresh, F, 0)
            
            # Count the new number of nonzero coefficients
            new_nonzero += np.count_nonzero(abs(F))
            
            # Re-introduce DC to matrix
            F[0,0] = F_0
            
            # Convert back from frequency domain
            result[i*32:(i+1)*32, j*32:(j+1)*32] = np.real(np.fft.ifft2(F))
    
    # Calculate the coefficient drop rate
    drop = (old_nonzero - new_nonzero) / old_nonzero

    return result, drop


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'input_files',
        type=str,
        nargs='+',
        help="The image file(s) to be compressed"
    )
    args = parser.parse_args()

    for filename in args.input_files:
        try:
            grayscale_image = np.array(Image.open(filename).convert('L'))
            compressed_image, drop_rate = compress(grayscale_image, COEFFICIENT_TOLERANCE_VALUE)
            Image.fromarray(compressed_image).convert('L').save('compressed_' + filename)

            print('Success! {} compressed with a drop rate of {:.2f}'.format(filename, drop_rate))
        except Exception as err:
            print(err)


if __name__ == "__main__":
    main()
