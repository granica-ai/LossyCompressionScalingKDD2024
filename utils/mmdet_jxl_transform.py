"""JPEG-XL Transform for mmdetection and mmsegmentation."""

from imagecodecs import jpegxl_encode, jpegxl_decode
from mmcv.transforms import BaseTransform, TRANSFORMS


@TRANSFORMS.register_module()
class JXLTransform(BaseTransform):
    def __init__(
        self, distance: float, distance_map: dict = None, path_key: str = "img_path"
    ):
        """
        Initialize the JPEG XL transformation.

        Args:
            distance (float): Butteraugli distance target for JPEG XL compression.
            distance_map (dict): Mapping from image path to Butteraugli distance.
            path_key (str): Key to access the image path in the results dictionary.
        """
        super().__init__()
        self.distance = distance
        self.distance_map = distance_map
        self.path_key = path_key

    def transform(self, results: dict) -> dict:
        """
        Apply the JPEG XL transformation to the input image.

        If the Butteraugli distance is set to 0, the input image is returned as is.

        Args:
            results (dict): Input image and metadata

        Returns:
            dict: Output image after lossy compression and decompression
        """
        if self.distance_map is not None:
            distance = self.distance_map[results[self.path_key].split("/")[-1]]
        else:
            distance = self.distance

        if distance == 0:
            return results
        img = results["img"]
        results["img"] = jpegxl_decode(jpegxl_encode(img, distance=distance))

        return results
