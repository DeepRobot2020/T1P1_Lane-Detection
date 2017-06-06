#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math
import os
import sys


def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=5):
    lines_left  = []
    lines_right = []

    # Iterate over the output "lines" and draw lines on a blank image
    for line in lines:
        for x1,y1,x2,y2 in line:
            y_delta = y2 - y1 
            x_delta = x2 - x1
            if y_delta/x_delta < 0:
                lines_left.append(line)
            else:
                lines_right.append(line)
    n = 1
    x1Avg = 0.0 
    x2Avg = 0.0 
    y1Avg = 0.0 
    y2Avg = 0.0 
    for line in lines_left:
        for x1,y1,x2,y2 in line:
            x1Avg = x1Avg + (x1 - x1Avg)/n
            x2Avg = x2Avg + (x2 - x2Avg)/n
            y1Avg = y1Avg + (y1 - y1Avg)/n
            y2Avg = y2Avg + (y2 - y2Avg)/n
            n += 1
    if x1Avg == 0.0 or y1Avg == 0.0 or x2Avg == 0.0 or y2Avg == 0.0:
        return
    [slope, intercept] = np.polyfit([x1Avg, x2Avg], [y1Avg, y2Avg], 1)
    startY = 330
    endY   = img.shape[0]
    startX = (startY - intercept) / slope
    endX = (endY - intercept) / slope
    cv2.line(img,(int(startX),int(startY)),(int(endX), int(endY)),color,thickness)

    n = 1
    x1Avg = 0.0 
    x2Avg = 0.0 
    y1Avg = 0.0 
    y2Avg = 0.0 

    for line in lines_right:
        for x1,y1,x2,y2 in line:
            x1Avg = x1Avg + (x1 - x1Avg)/n
            x2Avg = x2Avg + (x2 - x2Avg)/n
            y1Avg = y1Avg + (y1 - y1Avg)/n
            y2Avg = y2Avg + (y2 - y2Avg)/n
            n += 1
    if x1Avg == 0.0 or y1Avg == 0.0 or x2Avg == 0.0 or y2Avg == 0.0:
        return    
    [slope, intercept] = np.polyfit([x1Avg, x2Avg], [y1Avg, y2Avg], 1)
    startY = 330
    endY   = img.shape[0]
    startX = (startY - intercept) / slope
    endX = (endY - intercept) / slope
    cv2.line(img,(int(startX),int(startY)),(int(endX), int(endY)),color,thickness)



def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.    
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    if lines is None:
        print ('error, no hough lines are found')
        return None
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)

def process_image(image):
    # Define color selection criteria
    color_select    = np.copy(image)
    red_threshold   = 180
    green_threshold = 180
    blue_threshold  = 0
    rgb_threshold   = [red_threshold, green_threshold, blue_threshold]

    # Mask pixels below the threshold
    color_thresholds = (image[:,:,0] < rgb_threshold[0]) | \
                    (image[:,:,1] < rgb_threshold[1]) | \
                    (image[:,:,2] < rgb_threshold[2])

    # Mask color and region selection
    color_select[color_thresholds] = [0, 0, 0]

    gray = grayscale(color_select)
    # Define a kernel size and apply Gaussian smoothing
    kernel_size = 5
    blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)


    # Define our parameters for Canny and apply
    low_threshold = 50
    high_threshold = 150
    edges = canny(gray, low_threshold, high_threshold)

    # Next we'll create a masked edges image using cv2.fillPoly()

    imshape = image.shape
    vertices_outer = np.array([[ (150,imshape[0]), (450, 320), (550, 320),(890,imshape[0])]], dtype=np.int32)  
    vertices_inner = np.array([[ (190,imshape[0]), (450, 350), (550, 350),(700,imshape[0])]], dtype=np.int32)  

    vertices = np.concatenate((vertices_outer, vertices_inner), axis=0)
    roi = region_of_interest(edges, vertices)


    # Define the Hough transform parameters
    # Make a blank the same size as our image to draw on
    rho = 2 # distance resolution in pixels of the Hough grid
    theta = np.pi/180     # angular resolution in radians of the Hough grid
    threshold = 30         # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 50  # minimum number of pixels making up a line
    max_line_gap = 250     # maximum gap in pixels between connectable line segments
    line_image = hough_lines(roi, rho, theta, threshold, min_line_length, max_line_gap)
    if line_image is None:
        sys.exit(1)
        
    # Create a "color" binary image to combine with line image
    color_edges = np.dstack((edges, edges, edges)) 

    lines_edges = weighted_img(image, line_image, 0.8, 1, 0)
    return lines_edges


# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML



input_folder = 'test_videos/' 
output_folder = 'test_videos_output/' 

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for file in os.listdir(input_folder):
    print(file)
    print(os.path.isfile(file))
    processed_file = os.path.join(output_folder, file)
    clip = VideoFileClip(os.path.join(input_folder, file))
    white_clip = clip.fl_image(process_image) #NOTE: this function expects color images!! 
    white_clip.write_videofile(processed_file, audio=False)
    HTML("""
    <video width="960" height="540" controls>
    <source src="{0}">
    </video>
    """.format(processed_file))
