from algorithm.tools.extraction import *
from algorithm.tools.functions import *
import cv2
import numpy as np
import pywt


class Spatial_Domain:
    def __init__(self, img_input: np.ndarray) -> None:
        self.img_input =img_input
        self.nor_img = g_norm(img_input)

    def ela(self, compression: int, multp: int) -> np.ndarray:
        g_imageQuality(self.nor_img, 'ela_result.jpg', compression)
        compressed = cv2.imread('ela_result.jpg')
        diff = cv2.absdiff(self.nor_img, compressed)
        ela_img = np.clip(diff * multp, 0, 255).astype(np.uint8)
        extractEla(ela_img)


    def noise(self, local_mean: int, varience: int) -> None:
        local_gry = g_grayScale(self.img_input, np.float32)
        kernel = np.ones(local_mean, local_mean), np.float32

        # local variance
        mean = cv2.filter2D(local_gry, -1, kernel)
        mean_sq = cv2.filter2D(local_gry**2, -1, kernel)
        var = mean_sq - mean**2
        norm_var = g_normVariance(var, np.uint8)

        # PRNU
        prnu_gry = g_grayScale(self.nor_img, None)
        denoised = cv2.GaussianBlur(prnu_gry, (3, 3), 0)
        residual = prnu_gry - denoised
        prnu_map = g_normVariance(residual, np.uint8)

        # wavelet
        wavelet_gry = g_grayScale(self.img_input, np.float32)
        coeffs2 = pywt.dwt2(wavelet_gry, 'db1')
        LL,(LH, HL, HH) = coeffs2

        std_LH = np.std(LH)
        std_HL = np.std(HL)
        std_HH = np.std(HH)
        noise_estimation = (std_LH + std_HL + std_HH) / 3
        extractNoise(norm_var, prnu_map, noise_estimation)
    

    def copyMove(self, block_size: int, step: int, threshold: int) -> None:   
        norm_gry = g_grayScale(self.img_input)
        h, w = norm_gry.shape
        blocks = []
        position = []

        for y in range(0, h - block_size, step):
            for x in range(0, w - block_size, step):
                block = norm_gry[y:y + block_size, x:x + block_size]
                blocks.append(block.flatten())
                position.append((y, x))

        block = np.array(blocks)
        position = np.array(position)

        macthes = g_euclidenMaskPca(10, blocks, 1000)
        mask = np.zeros_like(norm_gry, dtype=np.uint8)
        for i, j in macthes:
           y1, x1 = position[i] 
           y2, x2 = position[j]
           cv2.rectangle(mask, (x1, y1), (x1 + block_size, y1 + block_size), 255, -1)
           cv2.rectangle(mask, (x2, y2), (x2 + block_size, y2 + block_size), 255, -1)
        
        extractCopyMove(mask)
    
    def resampling(self) -> None:
        resampling_gry = g_grayScale(self.img_input)
        mag = g_lapticanMagitude(resampling_gry)
        spectrum = g_normVariance(mag,np.uint8)
        extractResampling(spectrum)