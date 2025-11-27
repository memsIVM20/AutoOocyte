# This file for do a process of computer vision, when need to combine several function into one pipeline

import cv2
import csv
import numpy as np
import time
from datetime import datetime as dtime
import math
import os
import glob
import re
from scipy.signal import find_peaks
from ultralytics import YOLO

from camera_utilities.StartStreamCamera_Running import *                    # Camera library
from detect_main.preprocessing import imageProcessor                      # Image processor Class
from plc_communication import move_xyz                                      # PLC Control
from detect_main.spiral_gen import generate_spiral_and_valid_points       # Spiral Trajectory

# DB Numbers in PLC
motor_x = 1
motor_y = 2
motor_z = 3

pixel2mm = 1.4/864    #0.84     # Actual

# Global Variables
endthread = False 
confidence = 0.5

temp_m = 0.0
co2_m = 0.0
o2_m = 0.0

# Model Machine Learning YOLOv8
model = YOLO(r'detect_main\yolo\yolov8_ivm_new.pt')  # Path to the trained YOLO model

# Utilities
def deleteFiles(folder_path):
    '''
    Delete all the files in a directory

    '''
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        # Check if it is a file (not a folder)
        if os.path.isfile(file_path):
            os.remove(file_path)
            # print(f'Đã xóa file: {file_path}')

    return


# -----------------------------------------------------------------------------------------------
# ----------------------------------- Detect Well Edge ------------------------------------------
# -----------------------------------------------------------------------------------------------

def calculate_black_white_ratio(binary_image):
    '''
    Calculate the black to white (pixels) ratio in an image
    
    :param binary_image: A binary image (a threshold image)
    
    :return: The black to white (pixels) ratio
    
    '''

    # Make sure that the image is binary
    if len(binary_image.shape) != 2:
        raise ValueError("Input image is not binary.")

    # Counting the white pixels (value at 255) and the black pixels (value at 0)
    total_pixels = binary_image.size            # Total pixels in the image
    white_pixels = np.sum(binary_image == 255)  # Number of white pixels
    black_pixels = np.sum(binary_image == 0)    # Number of black pixels

    # Calculate the ratio
    white_ratio = white_pixels / total_pixels
    black_ratio = black_pixels / total_pixels

    # return white_pixels, black_pixels, white_ratio, black_ratio
    return black_ratio
    
def detectWellEdge(x_start, y_start, z_start, percentage= 0.3):
    '''
    Find the Well Edge
    
    :param x_start: The starting x coord
    :param y_start: The starting y coord
    :param percentage: The percentage of black pixels to white pixels needed to break the loop

    :return: The final x and y coord

    '''

    global endthread
    step_x = 0.1                        # Distance of each step in the x_axis
    x_scan = x_start                    # Assigning the x_scan the starting x coord
    allowed_distance = 100              # Maximum distance in mm

    # Moving all x, y and z axis to the starting position
    print("[detectWellEdge] MOVING TO START")
    move_xyz.move_coordination_motor_x(x_start)
    move_xyz.move_coordination_motor_y(y_start)
    move_xyz.move_coordination_motor_z(z_start)
    move_xyz.waiting_3_axis()

    print("[detectWellEdge] Starting MoveEdge X Axis")
    
    while True:

        if endthread == True:
            print("\n[detectWellEdge] BREAKING . . . ")
            break

        # Move X one step for each loop with a newly updated y_scan coord
        print("\n[detectWellEdge] MoveEdge X:", x_scan)
        move_xyz.move_coordination_motor_x(x_scan)
        move_xyz.waiting_1_axis(motor_x)
        
        image = imageProcessor(export_image())
           
        ratio = calculate_black_white_ratio(image.threshold_image2)
        print("[detectWellEdge] Ratio:", ratio*100, "%")
        
        if ratio > percentage :
            print(f"\n[detectWellEdge] Reached {percentage}% !!!")
            break
        else:
            x_scan += step_x            # Continue another moveEdge
            
        if x_start - x_scan >= allowed_distance:  #mm
            x_scan = x_start
            move_xyz.move_coordination_motor_x(x_scan)
            move_xyz.waiting_1_axis(motor_x)

            print("[detectWellEdge] Reached limit of x_scan. Returning to x_start....")
            break
        
    print("[detectWellEdge] MoveEdge DONE")

    x_out = x_scan      # Save the well position to use as a reference for the next search
    y_out = y_start

    return x_out, y_out

# detectWellEdge(x_starting_pos, y_starting_pos, z_starting_pos ratio)



# -----------------------------------------------------------------------------------------------
# ----------------------------------- Focus Well Edge -------------------------------------------
# -----------------------------------------------------------------------------------------------

def filter_contour(contours, threshold= 450):
    '''
    Filter out all the countours that do not meet the threshold requirement

    Input : contour list and threshold in arc curve length
    Output: the longest contour that satisfies the threshold

    '''
    
    longest_contour = None
    list_contour = []
    max_length = -10000
    ret = False
    
    for contour in contours:
        length = cv2.arcLength(contour, closed=False)
        if length > threshold :
            list_contour.append(contour)
            if length > max_length:
                longest_contour = contour
                max_length = length
                ret = True

    print(f'[filter_contour] Max length is {max_length}')
    
    return ret, list_contour, longest_contour   
    
def compute_variance_contour(mom_path):
    '''
    Input is a list of image and its peak index
    Output: list of variance of all peak image
    
    '''

    keep_list = []
    variance_list  = []

    for img_name in os.listdir(mom_path):
        keep_list.append(img_name)
        image = cv2.imread(os.path.join(mom_path,img_name))
        out_image = image.copy()

        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.GaussianBlur(gray, (5, 5), 0)  
            image = cv2.medianBlur(image, 5)
            threshold_value = 12  
            max_value = 255

            ret, threshold_image = cv2.threshold(gray, threshold_value, max_value, cv2.THRESH_BINARY)
            
            # Find all the contours in the filtered image
            canny_image = cv2.Canny(threshold_image, threshold1=30, threshold2=100)
            contours, _ = cv2.findContours(canny_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            ret, contours,_ = filter_contour(contours)

            for i, contour in enumerate(contours):
                # Compute the length of the contour
                length = cv2.arcLength(contour, closed=False)

                # Draw the contour on the image
                # cv2.drawContours(out_image, [contour], -1, (0, 255, 0), 2)  # Draw contours in green

                # Get the center point of the contour to position the text
                moments = cv2.moments(contour)
                if moments['m00'] != 0:  # To avoid division by zero
                    cx = int(moments['m10'] / moments['m00'])
                    cy = int(moments['m01'] / moments['m00'])
                else:
                    cx, cy = contour[0][0]  # If moments can't calculate center, use first point of contour

                # Step 4: Write the length on the image at the center of the contour
                cv2.putText(out_image, f"{length:.2f}", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            contour = max(contours, key=cv2.contourArea)  # Taking the largest contour
            contour_points = contour[:, 0, :]  # Extracting x, y positions

            # Split contour points into separate x and y arrays
            x = contour_points[:, 0]
            y = contour_points[:, 1]

            # Step 2: Fit a curve (e.g., 2nd degree polynomial) to the contour points
            coefficients = np.polyfit(x, y, deg=2)  # You can change 'deg' based on the trend
            poly_fit = np.poly1d(coefficients)

            # Step 3: Compute the expected y-values for each x using the fitted polynomial
            y_fitted = poly_fit(x)

            # Step 4: Calculate the residuals (difference between actual y and fitted y)
            residuals = y - y_fitted

            # Step 5: Compute the variance of the residuals
            variance = np.var(residuals)
        except:
            variance = 10000
        
        variance_list.append(variance)

        cv2.drawContours(out_image, contours, -1, (0, 255, 0), 2)
        cv2.imwrite(f'saved_images\contourApplied\{os.path.splitext(os.path.basename(img_name))[0]}_variance_{str(round(variance,3))}.jpg',out_image)
    
    return variance_list, keep_list

def delete_non_peak_image(mom_path, peak_list):
    '''
    Delete image files that are not a peak in a folder log.
    
    :param mom_path: Folder containing all images saved when moving the z axis, relative.
    :param peak_list: A list of all peaks index
    
    :return: A list of remaining images.
    
    '''
    
    keep_image_list = []

    for i in range(len(peak_list)):
        keep_image_list.append(os.path.join(f'img_{peak_list[i]}.jpg'))

    for file_name in os.listdir(mom_path):
    # Check if the file is not in the keep_images list
        if file_name not in keep_image_list:
            file_path = os.path.join(mom_path, file_name)
            
            # If the file is an image (you can filter by extension here if necessary)
            if os.path.isfile(file_path):
                # Delete the file
                os.remove(file_path)
                # print(f"Deleted: {file_name}")
            else:
                continue
            
    print(f"\n[delete_non_peak_image] Kept images: {keep_image_list}")
    return keep_image_list

def get_last_index_num(image_file):
    '''
    Get the index of the image
    
    '''
    file_name_without_ext = os.path.splitext(os.path.basename(image_file))[0]

    # Step 2: Use regular expressions to find the last number in the string
    last_number_str = re.findall(r'\d+', file_name_without_ext)[-1]

    # Step 3: Convert the last number to an integer
    last_number = int(last_number_str)
    
    return last_number
    # Compute variance of all peak image, then choose the lowest variable, return the index of peak has lowest variable

def tenengrad(img, ksize=3):
    '''' TENG' algorithm (Krotkov86) '''

    Gx = cv2.Sobel(img, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=ksize)
    Gy = cv2.Sobel(img, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=ksize)
    FM = Gx*Gx + Gy*Gy
    mn = cv2.mean(FM)[0]
    if np.isnan(mn):
        return np.nanmean(FM)
    
    return mn

def templateMatching(image, template, method=cv2.TM_CCOEFF_NORMED): # Insert a Grayscale image
    '''
    Calculate the image sharpness after circling the area using Template Matching
    
    :param image: A grayscaled image
    :param template: A grayscaled template image
    :param method: The Template Matching method
    
    :return: The sharpness value of the image
    
    '''
    h, w = template.shape               # Find the shape of the template image
    img2 = image.copy()                 # Create a second copy

    result = cv2.matchTemplate(img2, template, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    location = max_loc              # The upper left corner coords of the detected image
    bottom_right = (location[0] + w, location[1] + h)

    cropped_image = image[location[1]:bottom_right[1], location[0]:bottom_right[0]]             # Crop out the matching image
    
    tene = tenengrad(cropped_image)     # Calculate the Tenegrad value of the cropped image
    
    return tene

def focusWellEdge(z_start):
    '''
    Function to find the Z coord of the frame with the best focus

    Input : Starting Z coords
    Output: Final Z coords with the best Focus
    
    '''
    global endthread

    tene_list = []
    allowed_distance = 0.9    # Maximum distance that the z axis is allowed to travel (by mm)
    z_step = 0.01            # Distance of each step in the z axis
    z_scan = z_start

    if os.path.isfile("saved_images/log_tele.csv"):
        # Delete the file
        os.remove("saved_images/log_tele.csv")

    peak_tene_image_path = 'saved_images\log_image'
    contour_variance_image_path = 'saved_images\contourApplied'

    # Get a list of all files in the folder
    files = glob.glob(os.path.join(peak_tene_image_path, '*'))
    list(map(os.remove, files))
    files = glob.glob(os.path.join(contour_variance_image_path, '*'))
    list(map(os.remove, files))

    print("[focusWellEdge] All files log is deleted included csv, image in peak and variance contour \n")
    ######    ##################################################################################

    index = 0                 # Image index number
    
    template = cv2.imread("detect_main/template_pic/template_new.jpg")
    template_image = imageProcessor(template)       # Template Image
    
    while True:
        if endthread == True:
            print("\n[focusWellEdge] BREAKING . . .")
            break
        
        if(z_scan >= z_start + allowed_distance):   # Break if exceeds the allowed distance
            break
        
        image = imageProcessor(export_image())      # Replace with Camera read: DONE
        
        index += 1
        # if index < 10:
        #     continue

        tene = templateMatching(image.gray_image , template_image.gray_image)
        
        print("\n[focusWellEdge] Tene value is: ", tene)
        print(f'[focusWellEdge] Done Job: {round((z_start-z_scan) / allowed_distance * 100, 2) * (-1)} %')  # Print the searching progress
        
        z_scan += z_step
        move_xyz.move_coordination_motor_z(z_scan)              # Move command
        move_xyz.waiting_1_axis(motor_z)
        
        tene_list.append(tene)  # Create an array for the tene value of each step
        
        with open('saved_images/log_tele.csv', mode = 'a', newline = '') as file:
            writer = csv.writer(file)

            file.seek(0,2)
            if file.tell() == 0:                                                        # Save the data in a csv file and the image in a folder
                writer.writerow(['Index','Tenebraum','Z Height','Image path'])
            image_name = f'saved_images/log_image/img_{index}.jpg'
            writer.writerow([index, tene, z_scan, image_name])

            cv2.imwrite(image_name , image.original_image)

    if endthread == False:                    
        tene_list = np.array(tene_list)
        peaks, _ = find_peaks(tene_list, distance=60, height=60)                            # Find the peaks in the tene_list graph
        print("\n[focusWellEdge] Peaks are: ", peaks)
        
        if len(peaks) == 0:  # If there are no peaks
            print("\n[focusWellEdge] No peaks")
            max_index = np.argmax(tene_list)  # The index of the highest point
            peaks = [max_index]  # Assign the highest point as the peak
        
        _ = delete_non_peak_image('saved_images\log_image', peaks)                             # Delete (cut out) the images that are not peaks
        variance_list, keep_list_name = compute_variance_contour('saved_images\log_image')      
        print(f'\n[focusWellEdge] Keep file : {keep_list_name}\n')
        index_min_variance_contour = get_last_index_num(keep_list_name[np.argmin(variance_list)])   #  Get the index number of the image with the
                                                                                                    #  least variance of contours
        
        z_sharp_final = z_start + index_min_variance_contour * z_step                       # Calculate the final z coord with the sharpest image

        # time.sleep(0.2)
        move_xyz.move_coordination_motor_z(z_sharp_final)                                               # Move the camera to that z coord
        move_xyz.waiting_1_axis(motor_z)

        z_out = z_sharp_final
        return z_out , index_min_variance_contour
    
    else:
        return z_start, 1

# focusWellEdge(z_starting_pos)



# -----------------------------------------------------------------------------------------------
# ----------------------------------- Find Center -----------------------------------------------
# -----------------------------------------------------------------------------------------------

def circleApprox(A, B, C):
    """Estimate the center and radius of a circle passing through 3 given points.

    Args:
        A (tuple): First point (x1, y1)
        B (tuple): Second point (x2, y2)
        C (tuple): Third point (x3, y3)

    Returns:
        (tuple): (center_x, center_y), radius
    """

    print(f'\n[circleApprox] A: {A}, B: {B}, C: {C}')

    # Extract coordinates
    xa, ya = A
    xb, yb = B
    xc, yc = C
    
    # Denominator for determinant-based circle formula
    D = 2 * (xa * (yb - yc) + xb * (yc - ya) + xc * (ya - yb))
    
    if D == 0:
        raise ValueError("Points are collinear — cannot define a unique circle.")

    # Compute circle center coordinates
    Ux = ((xa**2 + ya**2) * (yb - yc) +
          (xb**2 + yb**2) * (yc - ya) +
          (xc**2 + yc**2) * (ya - yb)) / D
    
    Uy = ((xa**2 + ya**2) * (xc - xb) +
          (xb**2 + yb**2) * (xa - xc) +
          (xc**2 + yc**2) * (xb - xa)) / D
    
    # Calculate circle radius using distance from center to point A
    radius = math.sqrt((Ux - xa)**2 + (Uy - ya)**2)

    return (Ux, Uy), radius

def save_oocyte_with_circle(image, ct):
    ''' Save an image of a circle to monitor '''

    # print(f'ct before {ct}')
    
    # if len(ct) == 1:
    #     ct = ct[0]

    # Convert the contour to the correct shape (remove the unnecessary dimension)
    try:
        ct_max = ct.reshape(-1, 2)  # Reshape to (N, 2), where N is the number of points
    except:
        ct_max = ct

    # print(f'len ct after is {len(ct_max)}, ||| {ct_max}')

    image_color = image.copy()

    # Enlarge the frame
    offset_x_up = 500
    offset_y_up = 500
    offset_x_down = 3000
    offset_y_down = 3000

    h_origin, w_origin = image_color.shape[:2]
    new_size = (h_origin + offset_y_up + offset_y_down, w_origin + offset_x_up + offset_x_down, 3)

    new_img = np.zeros(new_size, dtype=np.uint8)  # Canvas image without content
    new_img[offset_y_up:h_origin + offset_y_up, offset_x_up:w_origin + offset_x_up] = image_color

    # Select points A, B, and C from the contour
    A_draw = ct_max[len(ct_max) // 10].copy()
    B_draw = ct_max[len(ct_max) // 5].copy()
    C_draw = ct_max[len(ct_max) // 2].copy()

    # Apply offset to A, B, and C
    A_draw[0] += offset_x_up
    A_draw[1] += offset_y_up

    B_draw[0] += offset_x_up
    B_draw[1] += offset_y_up

    C_draw[0] += offset_x_up
    C_draw[1] += offset_y_up

    # Calculate the center and radius using the circleApprox function
    print(f'\n[saveOocyte] Ba diem A, B, C de xap xi duoc chon lan luot la: {A_draw}, {B_draw}, {C_draw}')
    center, radius = circleApprox(A_draw, B_draw, C_draw)

    # Draw the center and the circle
    cv2.circle(new_img, center, 20, (255, 255, 255), thickness=2)  # Center

    # Draw the points A, B, C and annotate them
    cv2.circle(new_img, tuple(A_draw), 10, (255, 255, 0), thickness=10)
    cv2.putText(new_img, 'A', (A_draw[0] + 20, A_draw[1]), fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=5, color=(255, 255, 255), thickness=5)

    cv2.circle(new_img, tuple(B_draw), 10, (255, 255, 0), thickness=10)
    cv2.putText(new_img, 'B', (B_draw[0] + 20, B_draw[1]), fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=5, color=(255, 255, 255), thickness=5)

    cv2.circle(new_img, tuple(C_draw), 10, (255, 255, 0), thickness=10)
    cv2.putText(new_img, 'C', (C_draw[0] + 20, C_draw[1]), fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=5, color=(255, 255, 255), thickness=5)

    # Draw the circle with the calculated radius
    cv2.circle(new_img, center, int(radius), (255, 255, 255), thickness=10)

    # Put Radius on the image
    text = f"Radius {radius:.2f}"
    cv2.putText(new_img, text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 4.5  , (255, 255, 255), 4, cv2.LINE_AA)

    # Save the resulting image
    cv2.imwrite(f'saved_images/approxCircle/approxCircle_pixel.jpg', new_img)
    print(f'\n[saveOocyte] approxCircle SAVED with Radius: {radius:.2f}')

def findCenter(x_edge, y_edge, z_edge):
    '''
    Find the Center of the Well Edge (in mm)
    
    :param x_edge: Coord of x axis after finding Well Edge
    :param y_edge: Coord of y axis after finding Well Edge
    :param z_edge: Coord of z axis after finding Well Edge

    :return x_center_mm: Coord x of the Well Center in mm
    :return y_center_mm: Coord y of the Well Center in mm
    :return radius: Radius of the circle in mm
    
    '''
    
    global endthread

    ret = False                 # Return output
    radius = -1.0
    
    z_edge -= 0.0               # Offset the z axis
    move_xyz.move_coordination_motor_z(z_edge)
    move_xyz.waiting_1_axis(motor_z)

    for i in range(20):         # Try to find a suitable circle 20 times

        if endthread == True:
            print("\n[findCenter] BREAKING . . .")
            x_center_pixel = 1
            y_center_pixel = 1
            ret = False
            break

        image = imageProcessor(export_image())  # Replace with image Read: DONE

        cv2.imwrite(f'saved_images/MONITORING_1_original_img.jpg', image.original_image)
        cv2.imwrite(f'saved_images/MONITORING_2_gray_img.jpg', image.gray_image)
        cv2.imwrite(f'saved_images/MONITORING_3_adjusted_img.jpg', image.adjusted_image)
        cv2.imwrite(f'saved_images/MONITORING_4_gaublur_img.jpg', image.blurred_image)
        cv2.imwrite(f'saved_images/MONITORING_5_medblur_img.jpg', image.median_blur_image)
        cv2.imwrite(f'saved_images/MONITORING_6_threshold1_img.jpg', image.threshold_image)
        cv2.imwrite(f'saved_images/MONITORING_7_threshold2_img.jpg', image.threshold_image2)
        cv2.imwrite(f'saved_images/MONITORING_8_canny_img.jpg', image.canny_image)

        backup_image = image.original_image.copy()
        contours = image.contours

        if len(contours) < 1:
            x_center_pixel = 0
            y_center_pixel = 0
            radius = 0
            print(f"\n[findCenter] BREAKING . . . Could not find any contours")
            break

        ret_filter, contours, ct_max = filter_contour(contours)

        if ret_filter == False:
            x_center_pixel = 0
            y_center_pixel = 0
            radius = 0
            print(f"\n[findCenter] BREAKING . . . NULL CT_MAX")
            break
        
        out_image = image.original_image.copy()
        cv2.drawContours(out_image, contours, -1, (0, 255, 0), 2)

        cv2.imwrite(f'saved_images/MONITORING_8_sharpest_final_image_z.jpg', out_image)
        print("\n[findCenter] Saved the sharpest Final Contour Image")
            
        save_oocyte_with_circle(backup_image, ct_max) # save approximate circle on offset image 

        ct_max = np.squeeze(ct_max)
        A = ct_max[len(ct_max)//10]
        B = ct_max[len(ct_max)//5]
        C = ct_max[len(ct_max)//2]
        print(f'\n[findCenter] The three coord A, B, C approximately chosen: {A}, {B}, {C}')
        
        center_pixel ,radius = circleApprox(A,B,C)
        x_center_pixel, y_center_pixel = center_pixel

        if (radius > 1400 and radius < 1700) and (x_center_pixel > 0 and y_center_pixel > 0):           # Conditions for a suitable circle
            print(f"\n[findCenter] Found suitable radius: {radius}")
            print(f"[findCenter] Center_pixel is at: {center_pixel}")
            ret = True  # Return ret and break
            break
        else:
            print(f"\n[findCenter] Radius found is unsuitable: ---------------------------------------------- <<{round(radius,2)}>>")
    
    print(f'\n[findCenter] Center of the Spiral (pixel): ({round(x_center_pixel,2)},{round(y_center_pixel)}), radius la :{radius} pixel')
    print(f'[findCenter] Z_edge this time is: {z_edge}')

    # Move to the center Position
    dx = (x_center_pixel - 432)* pixel2mm  # 432 is half a frame along the x axis, the center of the cam
    dy = (y_center_pixel - 324)* pixel2mm  # 324 is half a frame along the y axis, the center of the cam

    x_center_mm = x_edge + dx 
    y_center_mm = y_edge + dy 
    
    if ret:
        move_xyz.move_coordination_motor_x(x_center_mm)
        move_xyz.move_coordination_motor_y(y_center_mm)
        move_xyz.waiting_2_axis(motor_x, motor_y)
        
        # time.sleep(0.1)

        z_edge += 0.6       # Offset the z axis
        move_xyz.move_coordination_motor_z(z_edge)
        move_xyz.waiting_1_axis(motor_z)
        print("[findCenter] Adjusted to đáy giếng level")   

    radius_mm = radius * pixel2mm
    z_out = z_edge
    print("[findCenter] Radius in mm: ", radius_mm)

    return ret, x_center_mm, y_center_mm, z_out, radius_mm

# findCenter(x_from_detectWellEdge, y_from_detectWellEdge, z_from_focusWellEdge)



# -----------------------------------------------------------------------------------------------
# ----------------------------------- Detect Oocyte ---------------------------------------------
# -----------------------------------------------------------------------------------------------

def get_bbox_center(x1, y1, x2, y2):
    """
    Compute the center coordinates of a bounding box.

    Args:
        x1 (float): Top-left x-coordinate.
        y1 (float): Top-left y-coordinate.
        x2 (float): Bottom-right x-coordinate.
        y2 (float): Bottom-right y-coordinate.

    Returns:
        tuple: (center_x, center_y) the center point of the bounding box.
    """
    
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    
    return center_x, center_y

def oocyteRealign():
    ''' Realign the frame to the oocyte '''
    
    global endthread

    x1 = -1                     # Set the negative value for the loop
    y1 = -1
    x2 = -1
    y2 = -1
    
    allowed_fails = 10
    count_timeout = 0
    ret = False

    while True:
        if endthread == True:
            print("\n[oocyteRealign] BREAKING . . .")
            break

        print("\n[oocyteRealign] Trying to find Oocyte...")
        image = export_image()
        results = model.predict(source = image, imgsz = 640, conf = confidence)

        for result in results:
            if len(result) > 0:
                print(f'\n[oocyteRealign] Detected {len(result)} circles')

                box = result.boxes[0]
                x1, y1, x2, y2 = box.xyxy[0]
                print("\n[oocyteRealign] Coordinates xyxy: ", x1, y1, x2, y2)

                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                print("[oocyteRealign] Coordinates xyxy NEW: ", x1, y1, x2, y2)

        if x1 >= 0 and y1 >= 0 and x2 >= 0 and y2 >=0:
            print("\n[oocyteRealign] Found the oocyte")
            ret = True
            break

        count_timeout += 1
        if count_timeout >= allowed_fails:     # Break if failed over x times
            print("\n[oocyteRealign] BREAKING . . . Did not find any oocyte")
            break
    
    if ret == True:
        x_center, y_center = get_bbox_center(x1,y1,x2,y2)
        x_now, y_now, z_now = move_xyz.read_current_position()

        dx = (x_center - 432)* pixel2mm
        dy = (y_center - 324)* pixel2mm

        x_center_mm = x_now + dx
        y_center_mm = y_now + dy

        time.sleep(0.2)
        move_xyz.move_coordination_motor_x(x_center_mm)
        move_xyz.move_coordination_motor_y(y_center_mm)
        move_xyz.waiting_2_axis(motor_x, motor_y)
        print("[oocyteRealign] Moved Oocyte to the middle of frame")

    return ret

def detectOocyte(x_center, y_center, z_edge, radius_mm):
    """
    Locate an oocyte by scanning the well area in a spiral pattern.

    Args:
        x_center (float): X coordinate of the spiral center.
        y_center (float): Y coordinate of the spiral center.
        z_edge (float): Z coordinate at the focused well edge.
        radius_mm (float): Scanning radius in millimeters.

    Returns:
        tuple: (x_true, y_true) - The final detected oocyte position in machine coordinates.
    """

    global endthread

    flag = False
    x_true = -1
    y_true = -1
    z_num = -1

    A = generate_spiral_and_valid_points(x_center, y_center)
    print(f"[detectOocyte] Number of points in spiral path: {len(A)}")

    # Directory to remove previously saved search images
    folder_path = f'saved_images/detectOocyte'
    deleteFiles(folder_path)

    allowed_length = len(A) - 1

    for i in range(0, allowed_length):

        if endthread is True:
            print("\n[detectOocyte] BREAKING due to stop request.")
            flag = False
            break
        
        x_scan, y_scan = A[i]
        print(f'[detectOocyte] Spiral scanning at: {(x_scan, y_scan)}')

        move_xyz.move_coordination_motor_x(x_scan)
        move_xyz.move_coordination_motor_y(y_scan)
        move_xyz.move_coordination_motor_z(z_edge)
        move_xyz.waiting_3_axis()
        time.sleep(0.1)
        
        results = model.predict(source=export_image(), imgsz=640, conf=confidence)

        for result in results:
            if len(result) > 0:
                print(f'\n[detectOocyte] Detected {len(result)} object(s)')
                flag = True

                box = result.boxes[0]
                x1, y1, x2, y2 = box.xyxy[0]
                x_center_bbox, y_center_bbox = get_bbox_center(x1, y1, x2, y2)
                print(f"[detectOocyte] Bounding box center: ({x_center_bbox}, {y_center_bbox})")

            annotated_frame = results[0].plot()
            cv2.imwrite(f'saved_images/SEARCH_boundingbox_image.jpg', annotated_frame)
            
        if flag is True:
            # Move detected oocyte to the center of the camera frame
            x_now, y_now, z_now = move_xyz.read_current_position()

            dx = (x_center_bbox - 432) * pixel2mm  # 432px: assumed frame center X
            dy = (y_center_bbox - 324) * pixel2mm  # 324px: assumed frame center Y

            x_center_mm = x_now + dx
            y_center_mm = y_now + dy

            time.sleep(0.2)
            move_xyz.move_coordination_motor_x(x_center_mm)
            move_xyz.move_coordination_motor_y(y_center_mm)
            move_xyz.waiting_2_axis(motor_x, motor_y)
            print("[detectOocyte] Oocyte moved to the camera center")

            done = oocyteRealign()
            flag = False

            if done is True:
                break

        print(f'[detectOocyte] Search progress: {round(i / allowed_length * 100, 2)} %')

    if flag is True:
        x_true = x_center_mm
        y_true = y_center_mm
    else:
        x_true = x_center
        y_true = y_center

    x_true, y_true, _ = move_xyz.read_current_position()

    return x_true, y_true


# detectOocyte(x_from_findCenter, y_from_findCenter, radius_from_findCenter)



# -----------------------------------------------------------------------------------------------
# ----------------------------------- Focus Oocyte ----------------------------------------------
# -----------------------------------------------------------------------------------------------

def imageCrop(image, x1, y1, x2, y2):
    ''' Crop an image '''
    cropped_image = image[ y1:y2 , x1:x2 ]
    return cropped_image

def focusOocyte(z_start):
    '''
    Find the best Focus Z Coord of the Oocyte
    
    :param z_start: Z Coord of the Oocyte

    :return: Z Coord of the Oocyte with the best Focus

    '''    

    global endthread

    time.sleep(0.5)

    x1 = -1                     # Set the negative value for the loop
    y1 = -1
    x2 = -1
    y2 = -1

    allowed_distance = 0.2     # Maximum distance that the z axis is allowed to travel (by mm)
    z_step = 0.005               # Distance of each step in the z axis
    z_scan = z_start - 0.2
    tene_max = 0
    index = 0

    folder_path = f'saved_images/bboxCropped'
    deleteFiles(folder_path)

    count_overtime = 0
    go_to_focus = True

    while True:
        if endthread == True:
            print("\n[focusOocyte] BREAKING . . .")
            break

        print("\n[focusOocyte] Trying to find Oocyte...")
        image = export_image()
        results = model.predict(source = image, imgsz = 640, conf = confidence)

        for result in results:
            if len(result) > 0:
                print(f'\n[focusOocyte] Detected {len(result)} circles')

                box = result.boxes[0]
                x1, y1, x2, y2 = box.xyxy[0]
                print("\n[focusOocyte] Tọa độ xyxy: ", x1, y1, x2, y2)

                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                print("[focusOocyte] Tọa độ xyxy NEW: ", x1, y1, x2, y2)

        if x1 >= 0 and y1 >= 0 and x2 >= 0 and y2 >=0:
            print("\n[focusOocyte] Found the oocyte")
            break
        
        count_overtime += 1

        if count_overtime >= 20:
            print("\n[focusOocyte] COULD NOT FIND OOCYTE")
            print("\n[focusOocyte] Breaking... ... ... ...")
            go_to_focus = False
            break

    while go_to_focus == True:
        if endthread == True:
            print("\n[focusOocyte] BREAKING . . .")
            break
        
        z_scan += z_step
        move_xyz.move_coordination_motor_z(z_scan)              # Move command
        move_xyz.waiting_1_axis(motor_z)
        
        image = export_image()

        bbox_crop_image_color = imageCrop(image, x1, y1, x2, y2)
        bbox_crop_image = cv2.cvtColor(bbox_crop_image_color, cv2.COLOR_BGR2GRAY)

        cv2.imwrite(f'saved_images/bboxCropped/boundingbox_image_{index}.jpg', bbox_crop_image_color)
        index += 1

        tene = tenengrad(bbox_crop_image)
        if tene > tene_max:
            tene_max = tene
            z_true = z_scan

        print("\n[focusOocyte] Tene value is: ", tene, "at", z_scan)
        print("[focusOocyte] Max tene is: ", tene_max, "at ", z_true)
        
        filename = "saved_images/tene_focusOocyte.csv"

        # Log data into a CSV file
        with open(filename, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([tene])

        if(z_scan >= z_start + allowed_distance):   # Break if exceeds the allowed distance
            break
        print(f'[focusOocyte] Done Job: {round((z_start-z_scan) / allowed_distance * 100, 2) * (-1)} %')  # Print the searching progress


    if go_to_focus == True:
        z_out = z_true
    else:
        z_out = z_start
        print("\n[focusOocyte] Did not find oocyte: Assigning z_start")

    print("\n[focusOocyte] FINAL z_sharpest:", z_out)
    move_xyz.move_coordination_motor_z(z_out)                                            # Move the camera to that z coord
    move_xyz.waiting_1_axis(motor_z)

    time.sleep(1)

    _, _, z_true = move_xyz.read_current_position()

    return z_out

# focusOocyte(z_from_focusWellEdge, T = 200)



# -----------------------------------------------------------------------------------------------
# --------------------------------- Stage 1: Find Edge ------------------------------------------
# -----------------------------------------------------------------------------------------------

def findEgde_Stage1(starting_positions, z_start, ratio= 0.3):
    ''' Stage One of the Process: Detect Well Edge, Focus Well Edge and Find Center '''

    global endthread

    well_index = 0
    
    for i in range(12):
        if endthread == True:
            print("\n[findEdge_Stage1] BREAKING . . .")
            break

        x_start = starting_positions[i][0]
        y_start = starting_positions[i][1]
        z_start = z_start

        print("\n\n----------------------------- DETECT WELL EDGE -------------------------------\n") 
        x_well, y_well = detectWellEdge(x_start=x_start, y_start=y_start, z_start=z_start, percentage=ratio)

        print("\n\n----------------------------- FOCUS WELL EDGE --------------------------------\n")
        z_focus, _ = focusWellEdge(z_start=z_start) 

        print("\n\n----------------------------- FIND CENTER ------------------------------------\n")
        ret, x_center, y_center, z_adjusted, radius = findCenter(x_edge=x_well, y_edge=y_well, z_edge=z_focus)

        print("\n[findEdge_Stage1] CENTER:", x_center, y_center)

        if ret == True:
            well_index = i
            print(f"\n[findEdge_Stage1] Found the center of TRAY 1 at well number: {i+1}")
            break
        else:
            print(f"\n[findEdge_Stage1] Could not find a suitable center for TRAY 1 at well number {i+1}")
            time.sleep(1)
    
    if endthread == False:
        return well_index, x_center, y_center, z_adjusted, radius
    else:
        return



# -----------------------------------------------------------------------------------------------
# ----------------------------------- Stage 2: Spiral -------------------------------------------
# -----------------------------------------------------------------------------------------------

oocytes_coord_1 = [[-1 for _ in range(3)] for _ in range(12)]       # Global lists for 12 wells coords
oocytes_coord_2 = [[-1 for _ in range(3)] for _ in range(12)]

def calculateOtherWells(well_index, x_center, y_center):            # Compute the remaining wells from the center of the first well (A1)
    wells_coord = []

    x_center_a1 = x_center - (well_index %  4) * 9
    y_center_a1 = y_center - (well_index // 4) * 9

    for index in range(12):

        x_index = (index %  4) * 9 + x_center_a1
        y_index = (index // 4) * 9 + y_center_a1

        wells_coord.append((x_index, y_index))

    return wells_coord

def calculateOtherWells_test(well_index, x_center, y_center, z_edge):            # Compute the remaining wells from the center of the first well (A1)
    wells_coord = []

    x_center_a1 = x_center - (well_index %  4) * 9
    y_center_a1 = y_center - (well_index // 4) * 9

    for index in range(12):

        x_index = (index %  4) * 9 + x_center_a1
        y_index = (index // 4) * 9 + y_center_a1

        wells_coord.append([x_index, y_index, z_edge])

    return wells_coord

def searchOtherWells_1(input_list, well_index, x_center_a1, y_center_a1, z_adjusted, radius):     # Search for oocyte in each selected well for Tray 1
    ''' TRAY 1 | Stage Two of the Process: Detect Oocyte and Focus Oocyte '''

    global endthread
    global oocytes_coord_1

    wells_coord = calculateOtherWells(well_index, x_center_a1, y_center_a1)
    
    folder_path = f'saved_images/finalOocyte_1'
    deleteFiles(folder_path)

    for index in input_list:
        if endthread == True:
            print("\n[searchOtherWells_1] BREAKING . . .")
            break

        if 1 <= index <= 12:  # Đảm bảo index nằm trong phạm vi hợp lệ
            x_true, y_true = detectOocyte(x_center= wells_coord[index-1][0], y_center= wells_coord[index-1][1], z_edge= z_adjusted, radius_mm= radius)
            z_true = focusOocyte(z_start= z_adjusted)

            image = imageProcessor(export_image())

            cv2.imwrite(f'saved_images/finalOocyte_1/detectedOocyte/final_Oocyte_{index}.jpg', image.original_image)

            oocytes_coord_1[index-1][0] = round(float(x_true),3)
            oocytes_coord_1[index-1][1] = round(float(y_true),3)
            oocytes_coord_1[index-1][2] = round(float(z_true),3)

            print("\n[searchOtherWells_1] Done Well number", index, "\n")

    print("\n[searchOtherWells_1] Finished All Wells")

    print("\n[searchOtherWells_1] List of final Coordinates: ", oocytes_coord_1)
    
    output_coord = oocytes_coord_1

    filename = "saved_images/final_coord_1.csv"

    # Ghi danh sách vào file CSV
    with open(filename, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerows(output_coord)

    print(f"\n[searchOtherWells_2] Data has been logged into: {filename}.")

    return output_coord

def searchOtherWells_2(input_list, well_index, x_center_a1, y_center_a1, z_adjusted, radius):       # Search for oocyte in each selected well for Tray 2
    ''' TRAY 2 | Stage Two of the Process: Detect Oocyte and Focus Oocyte '''
    
    global endthread
    global oocytes_coord_2

    wells_coord = calculateOtherWells(well_index, x_center_a1, y_center_a1)

    folder_path = f'saved_images/finalOocyte_2'
    deleteFiles(folder_path)
    
    for index in input_list:
        if endthread == True:
            print("\n[searchOtherWells_2] BREAKING . . .")
            break
        
        if 1 <= index <= 12:  # Đảm bảo index nằm trong phạm vi hợp lệ
            x_true, y_true = detectOocyte(x_center= wells_coord[index-1][0], y_center= wells_coord[index-1][1], z_edge= z_adjusted, radius_mm= radius)
            z_true = focusOocyte(z_start= z_adjusted)

            image = imageProcessor(export_image())

            cv2.imwrite(f'saved_images/finalOocyte_2/detectedOocyte/final_Oocyte_{index}.jpg', image.original_image)

            oocytes_coord_2[index-1][0] = round(float(x_true),3)
            oocytes_coord_2[index-1][1] = round(float(y_true),3)
            oocytes_coord_2[index-1][2] = round(float(z_true),3)

            print("\n[searchOtherWells_2] Done Well number", index, "\n")

    print("\n[searchOtherWells_2] Finished All Wells")

    print("\n[searchOtherWells_2] List of final Coordinates: ", oocytes_coord_2)
    
    output_coord = oocytes_coord_2

    filename = "saved_images/final_coord_2.csv"

    # Log data into a CSV file
    with open(filename, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerows(output_coord)

    print(f"\n[searchOtherWells_2] Data has been logged into: {filename}.")

    return output_coord


# -----------------------------------------------------------------------------------------------
# --------------------------------------- Timelapse ---------------------------------------------
# -----------------------------------------------------------------------------------------------

def processImageWithInfo(image, temp, co2, o2):
    n2 = 1000000 - co2 - o2

    font = cv2.FONT_HERSHEY_TRIPLEX
    font_scale = 0.55
    color = (0, 0, 0)  # Black (BGR format)
    thickness = 1

    now = dtime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")

    # Insert first line of text
    text0 = dt_string
    position0 = (20, 30)  # Below the first text
    cv2.putText(image, text0, position0, font, font_scale, color, thickness)

    text1 = f"Temp: {temp} oC"
    position1 = (20, 50)  # (x, y) position
    cv2.putText(image, text1, position1, font, font_scale, color, thickness)

    text2 = f"CO2:   {co2} ppm"
    position2 = (20, 70)  # Below the first text
    cv2.putText(image, text2, position2, font, font_scale, color, thickness)

    text3 = f"O2:    {o2} ppm"
    position3 = (20, 90)  # Below the first text
    cv2.putText(image, text3, position3, font, font_scale, color, thickness)

    text4 = f"N2:    {n2} ppm"
    position4 = (20, 110)  # Below the first text
    cv2.putText(image, text4, position4, font, font_scale, color, thickness)

    return image

def timelapse(input_list_tray1, oocyte_coords_tray1, input_list_tray2, oocyte_coords_tray2, duration):
    ''' Timelapse Process'''

    global oocytes_coord_1
    global oocytes_coord_2
    global temp_m, co2_m, o2_m

    z_step = 0.01
    num = 1

    if duration == 0:
        duration = 10**30

    start_time = time.time()
    while time.time() - start_time  < duration:
        if endthread == True:
            print("\n[timelapse] BREAKING . . .")
            break

        for index in input_list_tray1:
            if endthread == True:
                print("\n[timelapse] BREAKING . . .")
                break

            if 1<= index <= 12:
                x_coord = oocyte_coords_tray1[index-1][0]
                y_coord = oocyte_coords_tray1[index-1][1]
                z_coord = oocyte_coords_tray1[index-1][2]
                
                if x_coord > 0 and y_coord > 0 and z_coord > 0:
                    move_xyz.move_coordination_motor_x(x_coord)
                    move_xyz.move_coordination_motor_y(y_coord)
                    move_xyz.move_coordination_motor_z(z_coord)
                    move_xyz.waiting_3_axis()
                
                    x_new, y_new = detectOocyte(x_coord, y_coord, z_coord, 1.8)
                    oocyteRealign()

                    oocyte_coords_tray1[index-1][0] = x_new     # Update the new coordinates
                    oocyte_coords_tray1[index-1][1] = y_new

                    image = imageProcessor(export_image())
                    image = processImageWithInfo(image.adjusted_image, temp_m, co2_m, o2_m)
                    # cv2.imwrite(f'data/finalOocyte_1/timelapseOocyte_{index}/timelapse_{num}.jpg', image.adjusted_image)   # UYN

                    for i in range (-5,6):
                        z_scan = z_coord + i * (z_step)

                        move_xyz.move_coordination_motor_z(z_scan)
                        move_xyz.waiting_1_axis(motor_z)

                        image = imageProcessor(export_image())
                        image_save = processImageWithInfo(image.original_image, temp_m, co2_m, o2_m)

                        cv2.imwrite(f'data/finalOocyte_1/timelapseOocyte_{index}/timelapse_loop{num}_iter{i}.jpg', image_save)   # UYN

                    

                    print(f"\n[timelapse] LOOP #{num} | TRAY 1 | Taking photo of oocyte in well number {index}\n")

                time.sleep(1)

        for index in input_list_tray2:
            if endthread == True:
                print("\n[timelapse] BREAKING . . .")
                break

            if 1<= index <= 12:
                x_coord = oocyte_coords_tray2[index-1][0]
                y_coord = oocyte_coords_tray2[index-1][1]
                z_coord = oocyte_coords_tray2[index-1][2]

                if x_coord > 0 and y_coord > 0 and z_coord > 0:
                    move_xyz.move_coordination_motor_x(x_coord)
                    move_xyz.move_coordination_motor_y(y_coord)
                    move_xyz.move_coordination_motor_z(z_coord)
                    move_xyz.waiting_3_axis()
                    
                    x_new, y_new = detectOocyte(x_coord, y_coord, z_coord, 1.8)
                    oocyteRealign()

                    oocyte_coords_tray2[index-1][0] = x_new     # Update the new coordinates
                    oocyte_coords_tray2[index-1][1] = y_new

                    image = imageProcessor(export_image())
                    image = processImageWithInfo(image.adjusted_image, temp_m, co2_m, o2_m)
                    # cv2.imwrite(f'data/finalOocyte_2/timelapseOocyte_{index}/timelapse_{num}.jpg', image.adjusted_image)   # UYN

                    for i in range (-5,6):
                        z_scan = z_coord + i * (z_step)

                        move_xyz.move_coordination_motor_z(z_scan)
                        move_xyz.waiting_1_axis(motor_z)

                        image = imageProcessor(export_image())
                        image_save = processImageWithInfo(image.original_image, temp_m, co2_m, o2_m)
                        cv2.imwrite(f'data/finalOocyte_2/timelapseOocyte_{index}/timelapse_loop{num}_iter{i}.jpg', image_save)   # UYN

                    print(f"\n[timelapse] LOOP #{num} | TRAY 2 | Taking photo of oocyte in well number {index}\n")

                time.sleep(1)

        num += 1

    print(f"\n[timelapse] TIMELAPSE OVER: Duration of {time.time() - start_time} has exceeded")

    # Update the global lists with an updated list once the timelapse process is over
    oocytes_coord_1 = oocyte_coords_tray1       
    oocytes_coord_2 = oocyte_coords_tray2

    return

