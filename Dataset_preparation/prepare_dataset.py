import cv2
import numpy as np 
import os
import json
import shutil


def dilatation(img):
    kernel = np.ones((3,3), 'uint8')

    dilate_img = cv2.dilate(img, kernel, iterations=1)

    return dilate_img

def preprocess(img):
    image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    range_list = [  
                    [ [ 48, 222, 208], [ 68,242 ,288]],
                    [[48,208,198], [68,228,278]],

                    [[46,137,145], [66,157,225]],
                    [[48,212,196], [68,232,276]],
                   [[50,245,215], [70,265,295]],
                   [ [45,166,205] , [65,186,285] ],

                   [ [45,130,140] , [70,264,300] ],

                   [ [49,138,205] , [69,158,285] ],
                   [ [49,164,201] , [69,184,281] ],
                   [ [49,170,211] , [69,190,291] ],
                    [ [46,112,207] , [66,132,287] ],
                    [ [53,201,123] , [73,221,203] ],
[ [49,70,215] , [69,90,295] ],
[ [46,134,208] , [66,154,288] ],
[ [50,135,203] , [70,155,283] ],
[ [46,134,208] , [66,154,288] ],
[ [51,78,214] , [71,98,294] ],
[ [49,234,178] , [69,254,258] ],
[ [50,99,211] , [70,119,291] ],
[ [50,92,215] , [70,112,295] ],
[ [49,55,214] , [69,75,294] ],
[ [53,223,176] , [73,243,256] ],
[ [60,116,128] , [80,136,208] ],
[ [66,188,41] , [86,208,121] ],
[ [53,194,179] , [73,214,259] ],
[ [57,200,199] , [77,220,279] ],
[ [53,231,181] , [73,251,261] ]











                 ]


    mask = np.zeros([image.shape[0],image.shape[1]], dtype=np.uint8)
    for item in range_list:
        min_red = np.array(item[0])
        max_red = np.array(item[1])
        print("Range",min_red, max_red)
        mask_= cv2.inRange(image, min_red, max_red)


        mask = mask+mask_

    mask  = dilatation(mask)
    cv2.imwrite("preprossed.jpg",mask)

    return mask

def get_contour(mask,change_color=True,return_mask=False):

    try:
        (_, contours, hierarchy) = cv2.findContours(image = mask, 
            mode = cv2.RETR_EXTERNAL,
            method = cv2.CHAIN_APPROX_SIMPLE)
        # CHAIN_APPROX_SIMPLE :  remove redunant contour points
    except:
        (contours, hierarchy) = cv2.findContours(image = mask, 
            mode = cv2.RETR_EXTERNAL,
            method = cv2.CHAIN_APPROX_SIMPLE)        

    
    print("First contour",len(contours), len(contours[0]))

  
    return contours

def get_multiple_contour(mask):
    #mask = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)

    ret,thresh = cv2.threshold(mask,127,255,0)

    contours_, hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

    cnt = sorted(contours_, key=cv2.contourArea, reverse = True)


    return cnt


def main(input_folder):

    output_img_folder = "dataset_/images/"
    output_mask_folder = "dataset_/masks/"

    if not os.path.exists(output_img_folder):
        os.makedirs(output_img_folder)
    if not os.path.exists(output_mask_folder):
        os.makedirs(output_mask_folder)

    for file in os.listdir(input_folder):
        input_file = input_folder+file

        filename,ext = os.path.splitext(file)

        if ext == ".jpeg" or ext == ".jpg" and filename[-1] != "_":

            labeled_image_file = input_folder+filename+"_"+ext

            if os.path.exists(labeled_image_file):

                img_file = input_folder+file
                labeled_img = cv2.imread(labeled_image_file)

                out_img_file = output_img_folder+file
                shutil.copy(img_file, out_img_file)

                print(img_file)
                
                blank_mask = np.zeros([labeled_img.shape[0],labeled_img.shape[1],3], dtype=np.uint8)

                preprossed = preprocess(labeled_img)

                contour = get_contour(preprossed)





                cv2.drawContours(blank_mask, contour, -1, (255, 255, 255),thickness=-1)

                contours_ = get_multiple_contour(preprossed)

                if len(contours_) > 1:
                    for i in range(1, len(contours_)):
                        print(len(contours_[i]))
                        if len(contours_[i]) < len(contours_[0])/2:

                            cv2.drawContours(blank_mask, [contours_[i]], -1, (0, 0, 0),thickness=-1)

                print("--",len(contours_))



                out_mask_file = output_mask_folder+file

                cv2.imwrite(out_mask_file, blank_mask)

            else:
                print("File {} Not exists".format(labeled_image_file))
 



input_folder = "/home/santoshadhikari/project/sorted_2/"
main(input_folder)

# out_img_dim = (320,320)