import cv2
from PIL import Image
import numpy as np 

def get_RGB_cloth(img,mask, change_color = True):

    biggest_contour , b_box = get_contour(mask, change_color= change_color)

    blank_img = np.zeros([img.shape[0],img.shape[1],3], dtype=np.uint8)

    blank_img = cv2.drawContours(blank_img, [biggest_contour], -1, (255, 255, 255), -1)



    full_img_mask_draw= blank_img.copy()

    cropped_mask = blank_img[b_box[1]:b_box[3], b_box[0]:b_box[2]]

    (thresh, cropped_mask) = cv2.threshold(cropped_mask, 100, 255, cv2.THRESH_BINARY)
    (thresh, full_img_mask_draw) = cv2.threshold(full_img_mask_draw, 100, 255, cv2.THRESH_BINARY)


    
    cropped_rgb = img[b_box[1]:b_box[3], b_box[0]:b_box[2]]


    mask_out = np.zeros([cropped_rgb.shape[0],cropped_rgb.shape[1],3], dtype=np.uint8)

    cropped_rgb_cloth =  np.where(cropped_mask, cropped_rgb, mask_out).astype(np.uint8)




    return cropped_rgb_cloth, cropped_mask, biggest_contour, b_box


def get_contour(mask,change_color=True,return_mask=False):
    
    if  change_color:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    (thresh, mask) = cv2.threshold(mask, 20, 255, 0)
    try:
        (_, contours, hierarchy) = cv2.findContours(image = mask, 
            mode = cv2.RETR_TREE,
            method = cv2.CHAIN_APPROX_SIMPLE)
    except:
        (contours, hierarchy) = cv2.findContours(image = mask, 
            mode = cv2.RETR_TREE,
            method = cv2.CHAIN_APPROX_SIMPLE)        

    contours_sizes= [(cv2.contourArea(cnt), cnt) for cnt in contours]
    biggest_contour = max(contours_sizes, key=lambda x: x[0])[1]
    (x,y,w,h) = cv2.boundingRect(biggest_contour)
    b_box = [x,y,x+w,y+h]

    if return_mask:
        return biggest_contour,b_box, mask

    return biggest_contour,b_box


def get_mask(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    (_, mask) = cv2.threshold(gray, 10, 255, 0)
    return mask 
    

def get_contour_mask(image, change_color= True):

    print(image.shape)

    if  change_color:
        print("here we are")
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    _, image = cv2.threshold(image, 5, 255, 0)

    biggest_contour , b_box = get_contour(image, change_color= False)

    blank_img = np.zeros([image.shape[0],image.shape[1],3], dtype=np.uint8)

    blank_img = cv2.drawContours(blank_img, [biggest_contour], -1, (255, 255, 255), -1)


    return blank_img
