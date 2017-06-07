# **Finding Lane Lines on the Road** 
---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


### Reflection

### 1. Pipeline Description. 
The major laneline detection pipeline is implement inside 'process_image()' function

My pipeline consisted of below steps. 
* Step 1,  Select pixels whose value is larger color_thresholds. As the color of lines are either while or yellow. 
* Step 2,  Convert the color-selected images to grayscale and apply Gaussian filter to smooth the noise 
* Step 3,  Canny edge detection was applied to the image got from Step 2
* Step 4,  Define a region which has 8 vertices. Line to be detected are inside this region 
* Step 5,  Hough transform was called to find the laneline we are interested in   

In order to draw a single line on the left and right lanes, draw_lines() function has been updated with below changes 
* Seperate left and right lines by its slop. The left line slop should large than 0 and the right one is less than 0
* Average the points of left and right line to get the left and right laneline. 
* Call 'np.polyfit' to get slopt and intercept of lanelines
* Define start and end point on the image and draw estimated laneline 


### 2. Potential shortcomings with current pipeline
There is a few potential shortcomings:
1. Depends camera positions, the vertices to select region of interest might have to be updated. 


### 3. Suggest possible improvements to your pipeline
1. The way of separate left and right lines got from hough transform is not perfect. I am using thresold '0' to seperate left and right line. 
This is based on the assumption that camera is mounted in the middle of car. Depends on the camera location, the slope of both left and right lanes can be both larger or smaller than zero. We need a better way to find the thresold.
2. Draw line function is also not perfect. I am using average all the points to get lanelines. Better way would be line extrapolation. 
2. Current pipeline does not work well with the changlle.mp4
