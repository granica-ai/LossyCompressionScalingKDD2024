"""JPEG XL transformation for preprocesing pipeline."""

import numpy as np
import PIL
from imagecodecs import jpegxl_encode, jpegxl_decode


class JPEGXLTransform:
    def __init__(self, distance: float = 100):
        """
        Initialize the JPEG XL transformation.

        Args:
            distance (float): Butteraugli distance target for JPEG XL compression. Defaults to 100.
        """
        self.distance = distance

    def __call__(self, pil_image: PIL.Image) -> PIL.Image:
        """
        Apply the JPEG XL transformation to the input image.

        If the Butteraugli distance is set to 0, the input image is returned as is.

        Args:
            pil_image (PIL.Image): Input image

        Returns:
            PIL.Image: Output image after lossy compression and decompression
        """
        if self.distance == 0:
            return pil_image

        # Convert PIL Image to numpy array
        image_np = np.array(pil_image)

        # Compress and decompress using JPEG XL
        compressed_image = jpegxl_encode(image_np, distance=self.distance)
        decompressed_image = jpegxl_decode(compressed_image)

        # Convert numpy array back to PIL Image
        decompressed_image = PIL.Image.fromarray(decompressed_image, "RGB")

        return decompressed_image

    def __repr__(self):
        return f"JPEGXLTransform(distance={self.distance})"
