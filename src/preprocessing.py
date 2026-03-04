import cv2
import numpy as np


class PreprocessingFilters:

    @staticmethod
    def original(image):
        return image.astype(np.float32) / 255.0

    @staticmethod
    def clahe(image):
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return result.astype(np.float32) / 255.0

    @staticmethod
    def gaussian_sharpen(image):
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        sharpened = cv2.addWeighted(image, 1.5, blurred, -0.5, 0)
        sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
        return sharpened.astype(np.float32) / 255.0

    @staticmethod
    def bilateral(image):
        result = cv2.bilateralFilter(image, 9, 75, 75)
        return result.astype(np.float32) / 255.0

    @staticmethod
    def unsharp_mask(image):
        gaussian = cv2.GaussianBlur(image, (9, 9), 10.0)
        result = cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)
        result = np.clip(result, 0, 255).astype(np.uint8)
        return result.astype(np.float32) / 255.0

    @staticmethod
    def medical_enhanced(image):
        normalized = image.copy().astype(np.float32)
        for i in range(3):
            channel = normalized[:, :, i]
            p2, p98 = np.percentile(channel, (2, 98))
            normalized[:, :, i] = np.clip(
                (channel - p2) / (p98 - p2 + 1e-6) * 255, 0, 255
            )
        normalized = normalized.astype(np.uint8)

        lab = cv2.cvtColor(normalized, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        enhanced = cv2.bilateralFilter(enhanced, 5, 50, 50)

        gray = cv2.cvtColor(enhanced, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edges = cv2.dilate(edges, None, iterations=1)

        sharpening_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened = cv2.filter2D(enhanced, -1, sharpening_kernel)

        mask = edges.astype(np.float32) / 255.0
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        mask = np.stack([mask] * 3, axis=-1)

        result = (enhanced * (1 - mask * 0.3) + sharpened * mask * 0.3).astype(np.uint8)

        return result.astype(np.float32) / 255.0
