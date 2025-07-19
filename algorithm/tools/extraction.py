
from algorithm.tools.functions import *
import cv2
import numpy as np


def extractEla(ela_img: np.ndarray) -> None:
    gray = g_grayScale(ela_img)
    mean, std, max_error, total = g_msmtCalculation(gray)
    tamper_map = g_imageThreshold(gray, 200, 255, cv2.THRESH_BINARY)
    tamper_ratio  = g_imageRatio(float, np.sum, tamper_map, 255, gray.shape[0], gray.shape[1])

    g_saveImage([
        ('elaGray_output.jpg', gray),
        ('elaTm_output.jpg', tamper_map),
    ], remove_after=True)

    ela_extaction = {
        "mean_error": mean,
        "std_error": std,
        "max_error": max_error,
        "total_error": total,
        "tampered_ratio": tamper_ratio,
    }

    g_saveFeatures("Ela", ela_extaction)


def extractNoise(norm: np.ndarray, prnu: np.ndarray, wavelet: np.ndarray) -> None:
    mean_norm, std_norm, max_norm, total_norm = g_msmtCalculation(norm)
    mean_prnu, std_prnu, max_prnu, total_prnu = g_msmtCalculation(prnu)
    norm_thres = g_imageThreshold(norm, 200, 255)
    prnu_thres = g_imageThreshold(prnu, 200, 255)
    ratio_norm = g_imageRatio(float, np.sum, norm_thres, 255, norm.shape[0], norm.shape[1])
    ratio_prnu = g_imageRatio(float, np.sum, prnu_thres, 255, prnu.shape[0], prnu.shape[1])    

    g_saveImage([
        ("norm_output.jpg", norm),
        ("prnu_output.jpg", prnu),
        ("normThresh_output.jpg", norm_thres),
        ("prnuThresh_output.jpg", prnu_thres),
    ], remove_after=True)

    noise_extaction = {
        "mean_norm_error": mean_norm,
        "std_norm_error": std_norm,
        "max_norm_error": max_norm,
        "total_norm_error": total_norm,
        "local_norm_ratio": ratio_norm,

        "mean_prnu_error": mean_prnu,
        "std_prnu_error": std_prnu,
        "max_prnu_error": max_prnu,
        "total_prnu_error": total_prnu,
        "local_prnu_ratio": ratio_prnu,

        "wavelet": float(wavelet)

    }

    g_saveFeatures("Noise", noise_extaction)



def extractCopyMove(mask: np.ndarray) -> None:
    tamper_ratio = np.sum(mask) / 255 / (mask.shape[0] * mask.shape[1])
    total_tampered= int(np.sum(mask) / 255)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bbox_area = 0
    bbox_img = cv2.cvtColor(mask.copy(), cv2.COLOR_GRAY2BGR)

    if contours:
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        bbox_area = w * h
        cv2.rectangle(bbox_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        g_saveImage([
            ("copy_move_output.jpg", bbox_img),
        ], remove_after=True)

    copyMove_extaction = {
        "tamper_ratio": round(tamper_ratio),
        "tampered_area": total_tampered,
        "bounding_box": bbox_area,
        "tampered": tamper_ratio
    }

    g_saveFeatures("Copy-Move", copyMove_extaction)


def extractResampling(spectrum: np.ndarray) -> None:
    central_band = spectrum[
        spectrum.shape[0]//2 - 30 : spectrum.shape[0]//2 + 30,
        spectrum.shape[1]//2 - 30 : spectrum.shape[1]//2 + 30
    ]

    periodicity = float(np.std(central_band))
    g_saveImage([
        ("resample_output.jpg", spectrum)
    ], remove_after=True)

    resample_extaction = {
        "periodicity_std": round(periodicity, 4),
        "resampled": periodicity
        
    }

    g_saveFeatures("Resample", resample_extaction)
   