import cv2
import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from math import atan2, cos, sin, sqrt, pi

import skimage
from skimage.transform import resize
from skimage.io import imshow, imread, imshow_collection

#%matplotlib inline


class ProcessImage:
    
    def __init__(self, image_dir = None):
        self.image_dir = image_dir
        self.images = list()

    def read_single_image(self, file_name, clip = False):
        self.img = cv2.imread(file_name)
        #self.img = self.img[:,150:self.img.shape[1]-150]
        if clip:
            height, width = self.img.shape[:2]
            #image_size = height * width
            
            self.img = self.img[(height//9):(-height//9), :] 
        
        return  self.img.copy()
    
    
    def get_grid_image(self, image = None, xml_path = "/home/workstation/workspace/sifat_ahmed/concrete_detection/yolov5/model.yml.gz"):
        
        print("################", os.path.isfile(xml_path))
        # detector = cv2.ximgproc.createStructuredEdgeDetection(xml_path)
        


        # if image is None:
        #     image = self.get_gray().copy()
        
        # #gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # image =  detector.detectEdges(np.float32(image)/ 255)
        # image *= 255
        # image = image.astype(np.uint8)
        
        # return  image

    
    def get_gray(self, image = None):
        
        if image is None:
            self.gray_img = cv2.cvtColor(self.img.copy(), cv2.COLOR_BGR2GRAY)
            return self.gray_img
        else:
            return cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
    
    def equalize_hist(self, image = None):
        if image is None:
            return cv2.equalizeHist(self.image.copy())
        return cv2.equalizeHist(image.copy())
        
    def get_hsv(self, image = None):
        
        if image is None:
            self.hsv_img = cv2.cvtColor(self.img.copy(), cv2.COLOR_BGR2HSV)
            return self.hsv_img.copy()
        else: 
            return cv2.cvtColor(image.copy(), cv2.COLOR_BGR2HSV)
    
    
    def get_blurred(self, image = None, kernel_size = 3):
        
        if image is None:
            self.blur_img = cv2.GaussianBlur(self.img.copy(), (kernel_size, kernel_size),0)
            return self.blur_img.copy()
        else:
            return cv2.GaussianBlur(image.copy(), (kernel_size, kernel_size),0)
    
    def get_canny(self, threshold_low = 110, threshold_high = 210, image = None):
        
        if image is None:
            self.edges = cv2.Canny(np.uint8(self.img.copy()), threshold_low, threshold_high)
            return self.edges.copy()
        else:
            return cv2.Canny(np.uint8(image.copy()), threshold_low, threshold_high)
    
    def get_threshold(self,threshold_low = 110, threshold_high = 210, image = None):
        ret, thresh = cv2.threshold(self.get_gray(), threshold_low, threshold_high, 0)
        return thresh
    
    
    def get_binary_image(self, image = None):
        
        if image is None:
            image = self.img.copy()
        
        thresh, image =  cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return image
    
    
    
    def get_HoughLines(self,rho = 1, theta = np.pi/180, threshold = 30, line_length = 50, line_gap = 10, image = None, return_lines = False):
        
        rho = rho  # distance resolution in pixels of the Hough grid
        theta = theta  # angular resolution in radians of the Hough grid
        threshold = threshold  # minimum number of votes (intersections in Hough grid cell)
        min_line_length = line_length # minimum number of pixels making up a line
        max_line_gap = line_gap  # maximum gap in pixels between connectable line segments
        
        
        if image is None:
            line_image = np.copy(self.img.copy()) * 0
        else:
            line_image = np.copy(image.copy()) * 0
            
        # Run Hough on edge detected image
        # Output "lines" is an array containing endpoints of detected line segments
        lines = cv2.HoughLinesP(image.copy(), rho, theta, threshold, None, min_line_length, max_line_gap)

        for line in lines:
            for x1,y1,x2,y2 in line:
                cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),5)
        
        if return_lines:
            return line_image, lines
        
        return line_image
    
    
    def get_contours(self, apply_gray = False, image = None, threshold_low = 100, threshold_high = 200):
    
    
        if image is None:
            if apply_gray:
                ret, thresh = cv2.threshold(self.get_gray(), threshold_low, threshold_high, 0)
            else:
                ret, thresh = cv2.threshold(self.img.copy(), threshold_low, threshold_high, 0)
        else:
            if apply_gray:
                ret, thresh = cv2.threshold(self.get_gray(image.copy()), threshold_low, threshold_high, 0)
            else:
                ret, thresh = cv2.threshold(image.copy(), threshold_low, threshold_high, 0)    
        
        contours, hierarchy= cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        return contours
    
    
    def draw_contours_threshold(self, contours = None, max_area = 5000, height = 100, width = 100):
        if contours is None:
            raise("Missing contours")
        
        big_contours = list()
        temp_image = self.img.copy()
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            x, y, w, h = cv2.boundingRect(cnt)
            
            if area >= max_area and w >= width and h >= height:
                print(x,y,w,h, area)
                big_contours.append(cnt)
                
                temp_image = cv2.rectangle(temp_image, (x,y), (x+w, y+h), (0, 255, 0), 3)
        #cv2.drawContours(temp_image.copy(), big_contours, -1, (0,255, 0) ,thickness = cv2.FILLED)
        return temp_image
            
        
    def draw_contours(self, contours, image = None):
        
        if image is None:
            return cv2.drawContours(self.img.copy(), contours, -1, (0,255,0), thickness= cv2.FILLED)
        return cv2.drawContours(image.copy(), contours, -1, (0,255,0), thickness= cv2.FILLED)
    
    
    def  push_image_to_plot(self, image):
        self.images.append(image.copy())
    
    
    
    def apply_dilation(self, kernel_size = 5, iterations = 1 , image = None):
        
        if image is None:
            image = self.img.copy()
        
        kernel1 = np.ones((kernel_size, kernel_size), np.uint8)
        kernel2 = np.ones((kernel_size, kernel_size), np.uint8)
        
        #eroded = cv2.erode(image, kernel1, iterations = iterations)
        dialated = cv2.dilate(image, kernel2, iterations = iterations)
        
        return dialated
    

    def apply_erosion(self, kernel_size = 5, iterations = 1 , image = None):
        
        if image is None:
            image = self.img.copy()
            
        kernel1 = np.ones((kernel_size, kernel_size), np.uint8)
        eroded = cv2.erode(image, kernel1, iterations = iterations)
        
        return eroded
    
    def get_closing(self, image = None, kernel_size = 5, iterations = 2):
        if image is None:
            image = self.img.copy()
            
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations = iterations)
        
        return closing
    
    def get_opening(self, image = None, kernel_size = 5, iterations = 1):
        if image is None:
            image = self.img.copy()
            
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        closing = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations = iterations)
        
        return closing
    
    def show_image(self, image_list = None):
        
        if image_list is None:
            image_list = self.images
        
#         rows = int(np.ceil(len(image_list)/3))
#         cols = 3
        
#         count = 0
        
#         fig, axes = plt.subplots(nrows= rows, ncols=cols)
            
#         for i in range(rows):
#             for j in range(cols):
#                 if len(image_list) != 0:
#                     axes[i, j].imshow(image_list[count])
#                     count += 1
#                 else:
#                     #axes[i, j].imshow(self.image)
#                     break
        imshow_collection(self.images)
        plt.show()



def drawAxis(img, p_, q_, color, scale):
    p = list(p_)
    q = list(q_)

    ## [visualization1]
    angle = atan2(p[1] - q[1], p[0] - q[0]) # angle in radians
    hypotenuse = sqrt((p[1] - q[1]) * (p[1] - q[1]) + (p[0] - q[0]) * (p[0] - q[0]))

    # Here we lengthen the arrow by a factor of scale
    q[0] = p[0] - scale * hypotenuse * cos(angle)
    q[1] = p[1] - scale * hypotenuse * sin(angle)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 3, cv2.LINE_AA)

    # create the arrow hooks
    p[0] = q[0] + 9 * cos(angle + pi / 4)
    p[1] = q[1] + 9 * sin(angle + pi / 4)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 3, cv2.LINE_AA)

    p[0] = q[0] + 9 * cos(angle - pi / 4)
    p[1] = q[1] + 9 * sin(angle - pi / 4)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 3, cv2.LINE_AA)


def getOrientation(pts, img):
    ## [pca]
    # Construct a buffer used by the pca analysis
    sz = len(pts)
    data_pts = np.empty((sz, 2), dtype=np.float64)
    for i in range(data_pts.shape[0]):
        data_pts[i,0] = pts[i,0,0]
        data_pts[i,1] = pts[i,0,1]
 
    # Perform PCA analysis
    mean = np.empty((0))
    mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, mean)

    # Store the center of the object
    cntr = (int(mean[0,0]), int(mean[0,1]))
    ## [pca]

    ## [visualization]
    # Draw the principal components
    cv2.circle(img, cntr, 3, (255, 0, 255), 2)
    p1 = (cntr[0] + 0.02 * eigenvectors[0,0] * eigenvalues[0,0], cntr[1] + 0.02 * eigenvectors[0,1] * eigenvalues[0,0])
    p2 = (cntr[0] - 0.02 * eigenvectors[1,0] * eigenvalues[1,0], cntr[1] - 0.02 * eigenvectors[1,1] * eigenvalues[1,0])
    drawAxis(img, cntr, p1, (255, 255, 0), 1)
    drawAxis(img, cntr, p2, (0, 0, 255), 5)

    angle = atan2(eigenvectors[0,1], eigenvectors[0,0]) # orientation in radians
    ## [visualization]

    # Label with the rotation angle
    label = " " + str(-int(np.rad2deg(angle)) - 90) + " degrees"
    textbox = cv2.rectangle(img, (cntr[0], cntr[1]-25), (cntr[0] + 250, cntr[1] + 10), (255,255,255), -1)
    cv2.putText(img, label, (cntr[0], cntr[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)

    return angle


def get_edge(image):

    PI = ProcessImage()

    gray = PI.get_gray(image = image)
    #PI.push_image_to_plot(image = gray)

    edge_image = PI.get_grid_image(image = gray)

    # Convert image to binary
    bw = PI.get_binary_image(image = edge_image)
    return bw






