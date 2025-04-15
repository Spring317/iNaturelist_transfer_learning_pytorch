from torchvision import transforms  # type: ignore


class ColorDistorter:
    def __init__(self, ordering: int, brightness=31. / 255., contrast=0.5, saturation=0.5, hue=0.2):
        self.ordering = ordering
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue


    def __call__(self, img):
        ops = {
            -1: [self._brightness, self._saturation, self._hue, self._contrast],
            0: [self._saturation, self._brightness, self._contrast, self._hue],
            1: [self._contrast, self._hue, self._brightness, self._saturation],
            2: [self._hue, self._saturation, self._contrast, self._brightness]
        }
        for fn in ops[self.ordering % 3]:
            img = fn(img)
        return img


    def _brightness(self, img):
        return transforms.ColorJitter(brightness=self.brightness)(img)


    def _saturation(self, img):
        return transforms.ColorJitter(saturation=self.saturation)(img)


    def _contrast(self, img):
        return transforms.ColorJitter(contrast=self.contrast)(img)


    def _hue(self, img):
        return transforms.ColorJitter(hue=self.hue)(img)