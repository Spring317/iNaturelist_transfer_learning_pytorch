from typing import Tuple
from torchvision.transforms import functional  # type: ignore


class CentralCropResize:
    def __init__(self, central_fraction=0.875, size: Tuple[int, int]=(224, 224)):
        self.central_fraction = central_fraction
        self.size = list(size)


    def __call__(self, img):
        img = functional.to_tensor(img)
        _, h, w = img.shape
        crop_h = int(h * self.central_fraction)
        crop_w = int(w * self.central_fraction)
        top = (h - crop_h) // 2
        left = (w - crop_w) // 2

        img = functional.crop(img, top, left, crop_h, crop_w)
        img = functional.resize(img, self.size, interpolation=functional.InterpolationMode.BILINEAR)
        img = img.sub(0.5).mul(2.0)

        return img