import cv2
import sys
import os
import numpy as np 
from PIL import Image

cwd = os.getcwd()
sys.path.append("../yolov5/")
import inference_yolo


def create_3_channel(mask):
    img =  np.zeros([mask.shape[0],mask.shape[1],3], dtype=np.uint8)
    img[:,:,0] = mask # same value in each channel
    img[:,:,1] = mask
    img[:,:,2] = mask

    return img

def rgb_to_mask(image):
    blank =  np.zeros([image.shape[0], image.shape[1],3], dtype=np.uint8)

    contour, _ = get_contour(image,thresh_value=[5,255])
    cv2.drawContours(blank, [contour], -1, (255, 255, 255),thickness=-1)
    kernel = np.ones((3,   3 ), np.uint8)

    blank = cv2.erode(blank, kernel, iterations=1)

    cv2.imwrite("foreground_mask.jpg",blank)
    return blank 

def get_cropped_clean_mask(mask):# Removing small contour and performing erode and dilation to smooth edges
    blank_img = np.zeros([mask.shape[0],mask.shape[1],3], dtype=np.uint8)



    kernel = np.ones((5, 5), np.uint8)
 

    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)


    cnt, mask_box = get_contour(mask, change_color=False)
    cv2.drawContours(blank_img, [cnt], -1, (255, 255, 255),thickness=-1)

    cropped_mask = blank_img[mask_box[1]:mask_box[3], mask_box[0]:mask_box[2]]

    #cropped_mask =  create_3_channel(cropped_mask)

    return cropped_mask, blank_img, mask_box


def get_contour(mask,change_color=True, thresh_value=[210,255]):
    
    if  change_color:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    (thresh, mask) = cv2.threshold(mask, thresh_value[0], thresh_value[1], 0)
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

    return biggest_contour,b_box


def add_image(background, foreground, position):
    background_pil = Image.fromarray(background)
    foreground_pil  = Image.fromarray(foreground)

    background_pil.paste(foreground_pil,(int(position[0]),int(position[1])))

    backround_np =    np.asarray(background_pil)
    #backround_np = cv2.cvtColor(backround_np, cv2.COLOR_RGB2BGR)
    return backround_np


def overlay(foreground_mask, foreground, backround):
#    foreground_mask = rgb_to_mask(foreground)
    (thresh, foreground_mask) = cv2.threshold(foreground_mask, 200, 255, cv2.THRESH_BINARY)
    backround =  cv2.resize(backround,(foreground_mask.shape[1], foreground_mask.shape[0])) 
    added = np.where(foreground_mask, foreground, backround).astype(np.uint8) # foreground_mask, foreground_rgb, background_rgb

    return added

input_folder = "sorted_1/"
outfolder = input_folder[:-1]+"_out/"
if not os.path.exists(outfolder):
    os.makedirs(outfolder)

if not os.path.exists("debug"):
    os.makedirs("debug")

for file in os.listdir(input_folder):
    filename, ext = os.path.splitext(file)
    img_file = input_folder+file
    outfile = outfolder+file

    img = cv2.imread(img_file)
    original_input = img.copy()
    human_bbox = inference_yolo.main(img)
    print("before",human_bbox)
    if human_bbox is not None:
        human_bbox[0] = human_bbox[0]-20
        human_bbox[2] = human_bbox[2]+20
        
        if human_bbox[0]  <0:
           human_bbox[0]  = 0 

        if human_bbox[2]  > original_input.shape[1]:
           human_bbox[2]  = original_input.shape[1]
        print("after", human_bbox)

        sys.path.append("../U-2-Net/")
        sys.path.append("../pix2pixHD/")
        import inference_pix2pix
        import inference_unet
        print(human_bbox)
        cropped_image = img[int(human_bbox[1]):int(human_bbox[3]), int(human_bbox[0]):int(human_bbox[2])]
        cv2.imwrite("debug/cropped_human.jpg",cropped_image)
     

        mask = inference_unet.main(cropped_image)
        cv2.imwrite("debug/%s unet_mask.jpg"%filename,mask)
        blank_mask  = np.zeros([mask.shape[0],mask.shape[1],3], dtype=np.uint8)

        cropped_mask, _, mask_box = get_cropped_clean_mask(mask)

        cropped_mask_file = "debug/"+filename+"_cropped_mask.jpg"
        cv2.imwrite(cropped_mask_file,cropped_mask)

        translated, traslated_mask = inference_pix2pix.main(cropped_mask_file)
        cv2.imwrite("debug/translated.jpg", translated)
        cv2.imwrite("debug/translated_mask.jpg",traslated_mask)

        traslated_mask = rgb_to_mask(translated)
       
        translated = add_image(blank_mask, translated, mask_box)
        traslated_mask = add_image(blank_mask, traslated_mask, mask_box)


        overlayed = overlay(traslated_mask, translated, cropped_image)

        cv2.imwrite("debug/overlayed.jpg", overlayed)
        print(original_input.shape)
        original_overlayed = add_image(original_input,overlayed,human_bbox)

        cv2.imwrite(outfile,original_overlayed)
