# this file use for define some common function like filtering, coloring, bluring , etc. 

import cv2
import numpy as np
import math

def evaluateBrightnessAndAdjust(image, method="std", targetRange=(100, 200), maxIterations=3):
    """
    Evaluate the brightness of a grayscale image and adjust it iteratively if needed.

    :param image: Grayscale image (numpy array).
    :param method: Method for evaluation - "std", "his", "ev".
    :param targetRange: Tuple (min_brightness, max_brightness) defining the suitable range.
    :param maxIterations: Maximum number of iterations for adjustment.
    :return: Final adjusted image and evaluation results.
    """
    
    def calculateStd(image):
        # Standard deviation of pixel intensities
        return np.std(image)

    def calculateHis(image):
        # Histogram-based evaluation: Percentage of pixels within brightness thresholds
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        totalPixels = image.size
        lowThreshold, highThreshold = targetRange
        lowPixels = sum(hist[:lowThreshold])[0]
        highPixels = sum(hist[highThreshold:])[0]
        percentLow = (lowPixels / totalPixels) * 100
        percentHigh = (highPixels / totalPixels) * 100
        return percentLow, percentHigh

    def calculateEv(image):
        # Exposure value (EV): Combines mean and std deviation
        meanIntensity = np.mean(image)
        stdIntensity = np.std(image)
        if stdIntensity == 0:  # Prevent division by zero
            return float("inf")
        return math.log2(meanIntensity / stdIntensity)

    def adjustImageBrightness(image, alpha=1.0, beta=0):
        # Adjust brightness and contrast
        return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    def evaluateImage(image):
        # Evaluate brightness based on the selected method
        if method == "std":
            stdIntensity = calculateStd(image)
            return {"std": stdIntensity, "suitability": "low contrast" if stdIntensity < 30 else "suitable"}
        elif method == "his":
            percentLow, percentHigh = calculateHis(image)
            return {"percentLow": percentLow, "percentHigh": percentHigh,
                    "suitability": "too dark" if percentLow > 50 else "too bright" if percentHigh > 50 else "suitable"}
        elif method == "ev":
            ev = calculateEv(image)
            return {"ev": ev, "suitability": "underexposed" if ev < 1 else "overexposed" if ev > 3 else "balanced"}
        else:
            raise ValueError("Invalid evaluation method. Choose from 'std', 'his', or 'ev'.")

    # Initial evaluation
    evaluationResults = evaluateImage(image)
    adjustedImage = image
    iterations = 0

    # Iteratively adjust brightness until acceptable or maxIterations is reached
    while evaluationResults.get("suitability") in ["too dark", "underexposed", "low contrast", "too bright", "overexposed"] and iterations < maxIterations:
        if evaluationResults.get("suitability") in ["too dark", "underexposed", "low contrast"]:
            adjustedImage = adjustImageBrightness(adjustedImage, alpha=1.5, beta=50)  # Increase brightness and contrast
        elif evaluationResults.get("suitability") in ["too bright", "overexposed"]:
            adjustedImage = adjustImageBrightness(adjustedImage, alpha=0.95, beta=-5)  # Reduce brightness and contrast
        iterations += 1
        evaluationResults = evaluateImage(adjustedImage)  # Re-evaluate brightness

    return adjustedImage, evaluationResults

class imageProcessor:
    def __init__(self, image, method = "his", gau_kernel = 11, med_kernel = 5, thresh_val = 100, canny1 = 20, canny2 = 40):
        # Read image from the path
        self.original_image = image

        # Apply filters after creating an object
        self.gray_image = self.apply_grayscale(self.original_image)

        # Adjust brightness
        self.adjusted_image, _ = evaluateBrightnessAndAdjust(self.gray_image, method=method)

        self.blurred_image = self.apply_gaussian_blur(self.adjusted_image, kernel_size= (gau_kernel, gau_kernel))
        self.median_blur_image = self.apply_median_blur(self.blurred_image, kernel_size= med_kernel)
        self.threshold_image = self.apply_threshold(self.median_blur_image)
        self.threshold_image2 = self.apply_threshold2(self.median_blur_image, thresh_val= thresh_val)
        self.canny_image = self.apply_canny(self.adjusted_image, threshold1= canny1, threshold2= canny2)
        self.contours, self.hierarchy = self.find_contours(self.canny_image)
        
    def apply_grayscale(self, image):
        """Turn image into Grayscale."""
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def apply_gaussian_blur(self, image, kernel_size=(11, 11), sigma=1.5):
        """Apply Gaussian Blur."""
        return cv2.GaussianBlur(image, kernel_size, sigma)

    def apply_median_blur(self, image, kernel_size = 5):
        """Apply Median Blur."""
        return cv2.medianBlur(image, kernel_size)
    
    def apply_threshold(self, image, thresh_val= 0, max_val=255):
        """Apply Thresholding."""
        _, thresholded = cv2.threshold(image, thresh_val, max_val, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # _, thresholded = cv2.threshold(image, thresh_val, max_val, cv2.THRESH_BINARY )
        return thresholded
    
    def apply_threshold2(self, image, thresh_val= 255, max_val=255):
        """Apply Thresholding."""
        # _, thresholded = cv2.threshold(image, thresh_val, max_val, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, thresholded = cv2.threshold(image, thresh_val, max_val, cv2.THRESH_BINARY )
        return thresholded

    def apply_canny(self, image, threshold1=20, threshold2=130):
        """Apply Canny Edge Detection."""
        return cv2.Canny(image, threshold1, threshold2)

    def find_contours(self, image):
        """Find and return contours from the Canny Image."""
        contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours, hierarchy
    