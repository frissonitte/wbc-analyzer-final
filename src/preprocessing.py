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
        """Apply robust contrast normalization and edge-aware enhancement for smear images."""
        # Percentile clipping stabilizes exposure differences across microscopes.
        normalized = image.copy().astype(np.float32)
        for i in range(3):
            channel = normalized[:, :, i]
            p2, p98 = np.percentile(channel, (2, 98))
            normalized[:, :, i] = np.clip(
                (channel - p2) / (p98 - p2 + 1e-6) * 255, 0, 255
            )
        normalized = normalized.astype(np.uint8)

        # CLAHE in LAB space improves local contrast without shifting hue too much.
        lab = cv2.cvtColor(normalized, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        # Bilateral filter denoises while preserving boundaries.
        enhanced = cv2.bilateralFilter(enhanced, 5, 50, 50)

        gray = cv2.cvtColor(enhanced, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edges = cv2.dilate(edges, None, iterations=1)

        sharpening_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened = cv2.filter2D(enhanced, -1, sharpening_kernel)

        # Edge-weighted blending avoids over-sharpening smooth regions.
        mask = edges.astype(np.float32) / 255.0
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        mask = np.stack([mask] * 3, axis=-1)

        result = (enhanced * (1 - mask * 0.3) + sharpened * mask * 0.3).astype(np.uint8)

        return result.astype(np.float32) / 255.0

    @staticmethod
    def estimate_foreground_mask(image):
        """
        Estimate a soft foreground mask for the leukocyte region.
        The mask is only for XAI visualization and does not affect model predictions.
        """

        if image is None or image.size == 0:
            return np.ones((224, 224), dtype=np.float32)

        if image.dtype != np.uint8:
            if np.max(image) <= 1.0:
                image_u8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)
            else:
                image_u8 = np.clip(image, 0, 255).astype(np.uint8)
        else:
            image_u8 = image.copy()

        hsv = cv2.cvtColor(image_u8, cv2.COLOR_RGB2HSV)
        lab = cv2.cvtColor(image_u8, cv2.COLOR_RGB2LAB)

        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]
        a_channel = lab[:, :, 1]
        b_channel = lab[:, :, 2]

        sat_thr = max(20, int(np.percentile(saturation, 60)))
        val_thr = min(245, int(np.percentile(value, 85)))
        non_white = ((saturation > sat_thr) | (value < val_thr)).astype(np.uint8) * 255

        a_thr = int(np.percentile(a_channel, 60))
        b_thr = int(np.percentile(b_channel, 45))
        nucleus_hint = ((a_channel > a_thr) & (b_channel < b_thr)).astype(np.uint8) * 255

        mask = cv2.bitwise_or(non_white, nucleus_hint)

        # Morphological cleanup removes small background speckles.
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)
            h, w = mask.shape[:2]
            if area > 0.01 * h * w:
                clean = np.zeros_like(mask)
                cv2.drawContours(clean, [largest], -1, 255, thickness=-1)
                mask = clean

        soft_mask = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (0, 0), sigmaX=6, sigmaY=6)
        max_mask = np.max(soft_mask)
        if max_mask > 0:
            soft_mask = soft_mask / max_mask

        # If foreground estimation fails, return an all-ones mask to avoid hiding XAI.
        coverage = float(np.mean(soft_mask))
        if coverage < 0.05:
            return np.ones(mask.shape[:2], dtype=np.float32)

        return np.clip(soft_mask, 0.0, 1.0).astype(np.float32)
