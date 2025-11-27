# -- coding: utf-8 --
import sys
import _tkinter
import tkinter.messagebox
import tkinter as tk
import sys, os
import cv2 as cv
from tkinter import ttk

sys.path.append("../MvImport")

from camera_utilities.MvCameraControl_class import *
#from CamOperation_class import *
from camera_utilities.CameraConnectStream_Class import *
b_is_run = False

# Get the index of the selected device information by extracting the characters between two markers
def TxtWrapBy(start_str, end, all):
    start = all.find(start_str)
    if start >= 0:
        start += len(start_str)
        end = all.find(end, start)
        if end >= 0:
            return all[start:end].strip()

# Convert return error code to hexadecimal string
def ToHexStr(num):
    chaDic = {10: 'a', 11: 'b', 12: 'c', 13: 'd', 14: 'e', 15: 'f'}
    hexStr = ""
    if num < 0:
        num = num + 2**32
    while num >= 16:
        digit = num % 16
        hexStr = chaDic.get(digit, str(digit)) + hexStr
        num //= 16
    hexStr = chaDic.get(num, str(num)) + hexStr   
    return hexStr

def open_device():
        global deviceList
        global nSelCamIndex
        global obj_cam_operation
        global b_is_run
        if True == b_is_run:
            # tkinter.messagebox.showinfo('show info','Camera is Running!')
            return
        obj_cam_operation = CameraOperation(cam,deviceList,nSelCamIndex)
        ret = obj_cam_operation.Open_device()
        if  0!= ret:
            b_is_run = False
        else:
            #model_val.set('continuous')
            b_is_run = True
            
# Start grabbing images
def start_grabbing():
    global obj_cam_operation
    global stFrameInfo  
    global img_buff 
    obj_cam_operation.Start_grabbing()

# Stop grabbing images
def stop_grabbing():
    global obj_cam_operation
    obj_cam_operation.Stop_grabbing()    

# Close device   
def close_device():
    global b_is_run
    global obj_cam_operation
    obj_cam_operation.Close_device()
    b_is_run = False 
    
# Save jpg image
def jpg_save():
    global obj_cam_operation
    obj_cam_operation.b_save_jpg = True
        
# Link dropdown list selection to device information index
def xFunc():
    global nSelCamIndex
    nSelCamIndex = 0

# Enumerate devices
def enum_devices():
    global deviceList
    global obj_cam_operation
    deviceList = MV_CC_DEVICE_INFO_LIST()
    tlayerType = MV_GIGE_DEVICE | MV_USB_DEVICE
    ret = MvCamera.MV_CC_EnumDevices(tlayerType, deviceList)
    if ret != 0:
        print('enum devices fail! ret = '+ ToHexStr(ret))
    if deviceList.nDeviceNum == 0:
        print('find no device!')
        
    print ("Find %d devices!" % deviceList.nDeviceNum)
    devList = []
    mvcc_dev_info = cast(deviceList.pDeviceInfo[0], POINTER(MV_CC_DEVICE_INFO)).contents
    if mvcc_dev_info.nTLayerType == MV_USB_DEVICE:
            print ("\nu3v device: [%d]" % 0)
            strModeName = ""
            for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chModelName:
                if per == 0:
                    break
                strModeName = strModeName + chr(per)
            print ("device model name: %s" % strModeName)

            strSerialNumber = ""
            for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chSerialNumber:
                if per == 0:
                    break
                strSerialNumber = strSerialNumber + chr(per)
            print ("user serial number: %s" % strSerialNumber)
            devList.append("USB["+str(0)+"]"+str(strSerialNumber))

def startcamera():
    global deviceList 
    deviceList = MV_CC_DEVICE_INFO_LIST()
    global tlayerType
    tlayerType = MV_GIGE_DEVICE | MV_USB_DEVICE
    global cam
    cam = MvCamera()
    global nSelCamIndex
    nSelCamIndex = 0
    global obj_cam_operation
    obj_cam_operation = 0
    global b_is_run
    b_is_run = False
    
    # window = tk.Tk()
    # window.title('Stream Camera')
    # window.geometry('300x180')
    # #model_val = tk.StringVar()
    # global triggercheck_val
    # triggercheck_val = tk.IntVar()

    enum_devices()
    open_device()
    start_grabbing()

def export_image():
    image = obj_cam_operation.Export_image()
    return cv.resize(image, (864, 648))