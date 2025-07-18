from algorithm.others.extend_functions import *
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances
from typing import Union
from scipy.fftpack import fft2, fftshift
import os
import cv2
import shutil
import json
import numpy as np



def g_norm(img:np.ndarray) -> np.ndarray:
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    y_eq = cv2.equalizeHist(y)
    ycrcb_eq = cv2.merge([y_eq, cr, cb])
    normalized = cv2.cvtColor(ycrcb_eq, cv2.COLOR_YCrCb2BGR)
    return normalized

def g_normVariance(varience: int, dtype: type = None) -> np.ndarray:
    norm_var = cv2.normalize(varience, None, 0, 255, cv2.NORM_MINMAX)
    return norm_var.astype(dtype) if dtype is not None else norm_var

def g_grayScale(img: np.ndarray, dtype: type = None ) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray.astype(dtype) if dtype is not None else gray

def g_imageQuality(img: np.ndarray, compress: int) -> np.ndarray:
    cv2.imwrite('ela_result.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), compress])
    compress_img = cv2.imread('ela_result.jpg')
    g_saveImage([
        ('ela_result.jpg', compress_img)
    ], remove_after=True)
    return compress_img
    

def g_imageThreshold(img: np.ndarray, thresh_val: int, max_val: int, thresh_type: type= None) -> np.ndarray:
    if thresh_type is None:
        thresh_type = cv2.THRESH_BINARY
    _, thresholded = cv2.threshold(img, thresh_val, max_val, thresh_type)   
    return thresholded

def g_imageRatio(cast_type: type, sum_func: callable, img: np.ndarray, value: float, shape_height: int, shape_width: int) -> float:
    ratio = sum_func(img) / value / (shape_height * shape_width)
    return cast_type(ratio)

def g_msmtCalculation(img: np.ndarray) -> dict[str, int]:
    mean_error = float(np.mean(img))
    std_error = float(np.std(img))
    max_error = float(np.max(img))
    total_error = float(np.sum(img))

    return mean_error, std_error, max_error, total_error

def g_euclidenMaskPca(components_val: int, blocks: np.ndarray ,threshold: int) -> np.ndarray:
    pca = PCA(n_components=components_val)
    reduced = pca.fit_transform(blocks)

    dist = euclidean_distances(reduced)
    np.fill_diagonal(dist, np.inf)
    macthes = np.argwhere(dist < threshold)

    return macthes

def g_lapticanMagitude(img: np.ndarray) -> np.ndarray:

    laptican = cv2.Laplacian(img, cv2.CV_64F)
    abs_lap = np.abs(laptican)

    fft_img = fftshift(fft2(abs_lap))
    magnitude = np.abs(fft_img)
    magnitude = np.log1p(magnitude)

    return magnitude


# SAVE FUNCTIONS


def g_checkInput(extension: np.ndarray) -> bool:



    image_exts = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
    video_exts = (".mp4", ".avi", ".mov", ".mkv")
    
    ext = os.path.splitext(extension)[1].lower()

    if ext in image_exts:
        return True
    elif ext in video_exts:
        return False
    else:
        return "unsupported"


def g_saveImage(temp_imgs: Union[tuple[str, any], list[tuple[str, any]]], remove_after: bool = False) -> None:
    if isinstance(temp_imgs, tuple):
        temp_imgs = [temp_imgs]

    save_dir = os.path.join(
        "data",
        "output")
    os.makedirs(save_dir, exist_ok=True)

    for filename, img_data in temp_imgs:
        if not isinstance(filename, str):
            print(f"[CAUTION] INVALID FILENAME: {filename}")

        cv2.imwrite(filename, img_data)

        target_path = os.path.join(save_dir, os.path.basename(filename))
        shutil.copy(filename, target_path)

        if remove_after:
            os.remove(filename)


def g_saveFeatures(label: str, feature_data: dict, reset: bool = False):
    save_dir = os.path.join("data", "traning_data")
    os.makedirs(save_dir, exist_ok=True)
    json_path = os.path.join(save_dir, "features_data.json")

    # Handle existing file safely
    if reset or not os.path.exists(json_path):
        existing_data = {}
    else:
        try:
            with open(json_path, "r") as f:
                existing_data = json.load(f)
        except json.JSONDecodeError:
            print(f"[WARNING] Corrupted JSON detected at {json_path}. Starting fresh.")
            existing_data = {}

    existing_data[label] = feature_data

    # Save updated data
    with open(json_path, "w") as f:
        json.dump(existing_data, f, indent=4)

    print(f"'{label}' features saved to {json_path}")



def g_loadConfig() -> dict:
    config = extend_loadConfig()
    return config
