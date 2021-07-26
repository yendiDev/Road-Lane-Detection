import cv2
import numpy as np
import math

def region_of_interest(image, vertices):
    # create a mask of black color, get image channel count, and create match color for mask
    mask = np.zeros_like(image)
    # channel_count = image.shape[2]
    # match_mask_color = (255, )* channel_count
    match_mask_color = 255

    # extract region of interest from image, fill mask with roi, and return image
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


def draw_lines(image, lines):
    image_copy = image.copy()
    blank_image = np.zeros((image_copy.shape[0], image_copy.shape[1], 3), dtype=np.uint8)

    # loop through lines
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(blank_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # merge blank image with original image
    image = cv2.addWeighted(image, 0.8, blank_image, 1, gamma=0.0)
    return image

def process_frame(frame):
    # get image width and heigh
    width = frame.shape[1]
    height = frame.shape[0]

    # define region of interest
    region_of_interest_vertices = [
        (0, height),
        #(width/2 - 150, height/2 - 50),
        #(width/2, height/2 - 50),
        (width/2 , height/2 -100),
        (width, height)
    ]

    # convert cropped image to grayscale
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # apply canny detection on image
    canny_image = cv2.Canny(gray_image, threshold1=20, threshold2=90)

    # cropped image
    cropped_image = region_of_interest(canny_image, np.array([region_of_interest_vertices], dtype=np.int32),)

    # detect lines using probablistic hough transform
    lines = cv2.HoughLinesP(cropped_image, rho=6, theta=math.pi/60, threshold=160, lines=np.array([]), minLineLength=40, maxLineGap=40)

    # draw lines on image
    image_lines = draw_lines(frame, lines)

    return image_lines