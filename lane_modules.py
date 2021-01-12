import cv2
import numpy as np

def canny(image):
	# converting the image into grayscale
	gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
	# smoothing image using gaussian blur for noise reduction
	blur = cv2.GaussianBlur(gray, (5,5), 0)
	# applying canny edge detection methods
	canny = cv2.Canny(blur, 50, 150)
    
	return(canny)

def region_of_interest(image):
	# height is the number of rows
	height = image.shape[0]
	# specify region of interest
	polygons = np.array([
		[(200, height), (1100, height), (550, 250)]
		])
	# create mask image
	mask = np.zeros_like(image)
	# fill the ROI with white color
	cv2.fillPoly(mask, polygons, 255)
	# compute bitwise and operation between image and masked image
	masked_image = cv2.bitwise_and(image, mask)

	return masked_image

def make_coordinates(image, line_parameters):
    # define coordinates from line parameters
	slope, intercept = line_parameters
	y1 = image.shape[0]
	y2 = int(y1*(3/5))
	x1 = int((y1-intercept)/slope)
	x2 = int((y2-intercept)/slope)

	return np.array([x1, y1, x2, y2])

def average_slope_intercept(image, lines):
	"""
	Function to average the line images
	"""
	left_fit = []
	right_fit = []
	for line in lines:
		x1, y1, x2, y2 = line.reshape(4)
		# calculate the average lines using linear function (polyfit of degree 1)
		parameters = np.polyfit((x1,x2), (y1,y2), 1)
		slope = parameters[0]
		intercept = parameters[1]
        # if slope is negative, group image as left line
		if slope < 0:
			left_fit.append((slope, intercept))
        # if slope is positive, group image as right line
		else:
			right_fit.append((slope, intercept))
    # average the line matrix
	left_fit_average = np.average(left_fit, axis=0)
	right_fit_average = np.average(right_fit, axis=0)
    # map the coordinate into base image
	left_line = make_coordinates(image, left_fit_average)
	right_line = make_coordinates(image, right_fit_average)

	return np.array([left_line, right_line])

def hough_transform(image):
	"""
	Syntax : cv2. HoughLines(image, 
						resolution_x, 
						resolution_pi, 
						bin_size, 
						placeholder, 
						minLineLength, 
						maxLineGap)
	"""
	lines = cv2.HoughLinesP(image, 
						2, np.pi/280, 
						100, np.array([]),
						 minLineLength=40, 
						 maxLineGap=5)

	return lines

def display_lines(image, lines):
	"""
	Function to display black image
	overlayed with the lines detected
	by Hough Transform
	"""
    # create black image
	line_image = np.zeros_like(image)
	if lines is not None:
		for line in lines:
			# reshape the line image to 1D matrix  to get the x,y location of the lines
			x1, y1, x2, y2 = line.reshape(4)
			# draw line in the on the black image
			cv2.line(line_image, (x1,y1), (x2,y2), (255,0,0), 10)
	
	return line_image