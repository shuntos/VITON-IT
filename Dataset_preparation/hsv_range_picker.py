import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
import sys

##HSV color range picker
#Using Tkinter
#Shuntos-2018


ftypes = [
    ('JPG', '*.jpg;*.JPG;*.JPEG'), 
    ('PNG', '*.png;*.PNG'),
    ('GIF', '*.gif;*.GIF'),
]
image_hsv = None
pixel = (0,0,0) 

def pick_color(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
        pixel = image_hsv[y,x]

        #HUE, SATURATION, AND VALUE (BRIGHTNESS) RANGES. TOLERANCE COULD BE ADJUSTED.
        upper =  np.array([pixel[0] + 10, pixel[1] + 10, pixel[2] + 40])
        lower =  np.array([pixel[0] - 10, pixel[1] - 10, pixel[2] - 40])

        upper_str = "["
        for item in list(upper):
            upper_str += str(item)+","

        upper_str = upper_str[:-1]+"]"
        
        lower_str = "["
        for item in list(lower):
            lower_str += str(item)+","

        lower_str = lower_str[:-1]+"]"

        print("[",lower_str,",", upper_str,"]")

        #A MONOCHROME MASK FOR GETTING A BETTER VISION OVER THE COLORS 
        image_mask = cv2.inRange(image_hsv,lower,upper)
        cv2.imshow("Mask",image_mask)

def main():

    global image_hsv, pixel

    root = tk.Tk()
    root.withdraw() #HIDE THE TKINTER GUI
    image_src = cv2.imread("/home/santoshadhikari/project/sorted_2/1404_.jpg")
    cv2.imshow("BGR",image_src)

    #CREATE THE HSV FROM THE BGR IMAGE
    image_hsv = cv2.cvtColor(image_src,cv2.COLOR_BGR2HSV)
    cv2.imshow("HSV",image_hsv)

    #CALLBACK FUNCTION
    cv2.setMouseCallback("HSV", pick_color)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__=='__main__':
    main()
