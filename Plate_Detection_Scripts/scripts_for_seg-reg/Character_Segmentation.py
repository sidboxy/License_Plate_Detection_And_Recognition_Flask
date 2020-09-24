import numpy as np
import cv2


# Match contours to license plate or character template
def find_contours(dimensions, img):
    # Find all contours in the image
    image = img.copy()
    cntrs, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Retrieve potential dimensions
    lower_width = dimensions[0]
    upper_width = dimensions[1]
    lower_height = dimensions[2]
    upper_height = dimensions[3]

    # Check largest 5 or  15 contours for license plate or character respectively
    cntrs = sorted(cntrs, key=cv2.contourArea, reverse=True)[:15]


    x_cntr_list = []
    target_contours = []
    img_res = []
    for cntr in cntrs:
        # detects contour in binary image and returns the coordinates of rectangle enclosing it
        intX, intY, intWidth, intHeight = cv2.boundingRect(cntr)
        cv2.rectangle(image, (intX, intY), (intX + intWidth, intY + intHeight), (0, 0, 255), 2)

        # checking the dimensions of the contour to filter out the characters by contour's size
        if intWidth < upper_width and intHeight > lower_height and intHeight < upper_height:
            x_cntr_list.append(intX)  # stores the x coordinate of the character's contour, to used later for indexing the contours

            char_copy = np.zeros((28, 28))
            # extracting each character using the enclosing rectangle's coordinates.
            char = img[intY:intY + intHeight, intX:intX + intWidth]
            char = cv2.resize(char, (24, 24))

            # Make result formatted for classification: invert colors
            char = cv2.subtract(255, char)

            # Resize the image to 24x44 with black border
            char_copy[2:26, 2:26] = char
            char_copy[0:2, :] = 0
            char_copy[:, 0:2] = 0
            char_copy[26:28, :] = 0
            char_copy[:, 26:28] = 0

            img_res.append(char_copy)  # List that stores the character's binary image (unsorted)
    # Return characters on ascending order with respect to the x-coordinate (most-left character first)

    # arbitrary function that stores sorted list of character indeces
    indices = sorted(range(len(x_cntr_list)), key=lambda k: x_cntr_list[k])
    img_res_copy = []
    for idx in indices:
        img_res_copy.append(img_res[idx])  # stores character images according to their index
    img_res = np.array(img_res_copy)
    cv2.imwrite('segmentation.jpg', image)

    return img_res

def segment_characters(image) :
    image = cv2.imread(image)
    # Preprocess cropped license plate image
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, img_binary = cv2.threshold(img_gray, 200, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    img_erode = cv2.erode(img_binary, (3,3))

    LP_WIDTH = img_erode.shape[0]
    LP_HEIGHT = img_erode.shape[1]


    # Estimations of character contours sizes of cropped license plates
    dimensions = [LP_WIDTH/6, LP_WIDTH/2, LP_HEIGHT/10, 2*LP_HEIGHT/3]

    # Get contours within cropped license plate
    char_list = find_contours(dimensions, img_erode)
    return char_list
