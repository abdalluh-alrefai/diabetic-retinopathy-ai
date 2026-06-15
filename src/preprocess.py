"""
Fundus image preprocessing.

The classic Ben Graham preprocessing (winner of the original 2015 Kaggle DR
competition) dramatically improves signal: it crops the black border, resizes,
and subtracts the local average colour to enhance vessels and lesions.
"""
import cv2
import numpy as np


def crop_black_borders(img: np.ndarray, tol: int = 7) -> np.ndarray:
    """Remove the surrounding black region of a fundus photo."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask = gray > tol
    if mask.sum() == 0:
        return img
    coords = np.ix_(mask.any(1), mask.any(0))
    return img[coords]


def ben_graham(img: np.ndarray, size: int = 384, sigma: int = 10) -> np.ndarray:
    """Ben Graham colour-normalisation + resize.

    Args:
        img: RGB uint8 image.
        size: output square size.
        sigma: Gaussian blur radius for background estimation.
    """
    img = crop_black_borders(img)
    img = cv2.resize(img, (size, size))
    blurred = cv2.GaussianBlur(img, (0, 0), sigma)
    img = cv2.addWeighted(img, 4, blurred, -4, 128)
    # circular mask to drop corner artifacts
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.circle(mask, (size // 2, size // 2), int(size * 0.49), 1, -1, 8)
    img = img * mask[..., None] + 128 * (1 - mask[..., None])
    return img.astype(np.uint8)


def load_and_preprocess(path: str, size: int = 384) -> np.ndarray:
    """Read an image from disk and return a preprocessed RGB array."""
    bgr = cv2.imread(path)
    if bgr is None:
        raise FileNotFoundError(path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return ben_graham(rgb, size=size)


if __name__ == "__main__":
    import sys
    out = load_and_preprocess(sys.argv[1])
    cv2.imwrite("preprocessed.png", cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
    print("Saved preprocessed.png", out.shape)
