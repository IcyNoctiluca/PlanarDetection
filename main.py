''' SSA III CV assignment                                                   '''
''' Python 3.5.2    OpenCV 3.3.0                                            '''
''' Iterates through dataset, uses ransac to detect the eqn of the plane
    of the road. Logs the normal to the found plane to the console
    Highlights the boundary of road on a shown image
    Takes approx 15 secs per frame                                          '''
''' Some code here is based on Toby Breckon's stero_disparity.py and
    stereo_to_3d.py                                                         '''


# Importing pkgs & libs
import cv2
import math
import numpy as np
import os
import random


# Set up dir structure and others
masterPath = "/home/liam/Desktop/SSA/CVision/Assignment/TTBB-durham-02-10-17-sub10"
#masterPath = "J:\CV\TTBB-durham-02-10-17-sub10"
leftDirectory = "left-images"
rightDirectory = "right-images"

leftPath =  os.path.join(masterPath, leftDirectory)
rightPath =  os.path.join(masterPath, rightDirectory)
leftFileList = sorted(os.listdir(leftPath))


''' Set of functions to complement
    main flow                     '''

# Return the processed image before sending to disparity matching
def preprocessImg(img):

    # duplicate image for manipulation
    newImg = img.copy()
    newImg = convertGray(newImg)

    # expand range of pixel values
    newImg = cv2.equalizeHist(newImg)

    #kernalSize = 3
    #newImg = cv2.GaussianBlur(newImg, (kernalSize, kernalSize), 0)

    # sharpen the image to enchance disparity matching
    kern = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    #kern = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
    newImg = cv2.filter2D(newImg, -1, kern)

    return newImg


# Finds all dominant lines (road markings usually)
# not used in main script, was experimented with but shown to be unreliable
def lineFinder(img):

    # get one image to use for heurstistics to find road lines and one to draw the lines
    lineHeuristicsImg, drawLinesImg = convertGray(img.copy()), convertGray(img.copy())

    # equalise to enhance line detection
    lineHeuristicsImg, drawLinesImg = cv2.equalizeHist(lineHeuristicsImg), cv2.equalizeHist(drawLinesImg)

    # remove noise
    kernalSize = 7
    lineHeuristicsImg = cv2.GaussianBlur(lineHeuristicsImg, (kernalSize, kernalSize), 0)

    # get edges
    lineHeuristicsImg = cv2.Canny(lineHeuristicsImg, threshold1=50, threshold2=250)

    # edges for only ROI to reduce computation
    lineHeuristicsImg = getROI(lineHeuristicsImg)

    # get major lines in image
    lines = cv2.HoughLinesP(lineHeuristicsImg, 1, np.pi/180, 180, np.array([]), 200, 100)

    # plot lines on the draw image
    try:
        for line in lines:
            coords = line[0]
            cv2.line(drawLinesImg, (coords[0], coords[1]), (coords[2], coords[3]), [0, 0, 0], 10)
    except:
        pass

    return drawLinesImg, lineHeuristicsImg


# Returns coords of points which are within heightThresh of the roadPlane
def closeToRoad(Xs, Ys, Zs, roadPlane, heightThresh):

    # Check coord arrays are all same length
    if len(Xs) == len(Ys) == len(Zs):
        pass
    else:
        raise ValueError('Coord arrays are not same length :/')


    # reshape
    length = len(Xs)
    Xs, Ys, Zs = Xs.reshape((length, 1)), Ys.reshape((length, 1)), Zs.reshape((length, 1))

    # setup array to hold coords
    roadXs, roadYs, roadZs = np.array([]), np.array([]), np.array([])


    # Check whether coord is within the threshold (e.g. 0.05m of height of the plane of the road)
    for i in range (len(Xs)):

        # calculate plane height for each x and z coord
        y = (roadPlane[3] - roadPlane[0] * Xs[i] - roadPlane[2] * Zs[i]) / float(roadPlane[1])

        # difference between height of coord and plane hieght given X and Z positioning
        error = abs(y - Ys[i])

        # if points is within the threshold, deem to be close to plane
        if error < heightThresh:
            roadXs = np.append(roadXs, Xs[i])
            roadYs = np.append(roadYs, Ys[i])
            roadZs = np.append(roadZs, Zs[i])

    return roadXs, roadYs, roadZs



# Return the image with non regions of interest in black
# ROI defined by preset coords, adaptive ROI by lineFinder(img) proved too unreliable :'(
def getROI(img):

    #set up duplicate for manipulation
    newImg = img.copy()
    newImg = convertGray(newImg)

    # array of black in shape of img
    h, w = np.shape(newImg)
    blanck = np.zeros(h * w).reshape((h, w)).astype(np.uint8)

    # coords defining ROI in clockwise direction around the polygon
    botLeft, midLeft, peakLeft, peakRight, midRight, botRight, botMiddle = [0, h], [0, h/2], [w/3, h/4], [2*w/3, h/4], [w, h/2], [w, h], [w/2, 3*h/4]
    roiBounds = np.array([botLeft, midLeft, peakLeft, peakRight, midRight, botRight, botMiddle], np.int32)

    cv2.fillPoly(blanck, [roiBounds], 255)

    # Black out the areas outside the ROI specified by coords
    ROI = cv2.bitwise_and(newImg, blanck)

    return ROI


# returns contour for points deemed close to road. I.e. project 3D coords to 2D image
def findContour(roadXs, roadYs, roadZs, h, w):

    # Check coord arrays are all same length
    if len(roadXs) == len(roadYs) == len(roadZs):
        pass
    else:
        raise ValueError('Coord arrays are not same length :/')


    focalLengthPX = 399.9745178222656
    centreH = 262.0
    centreW = 474.5

    # set up blank to scatter points deemed close to road
    black = np.zeros(h * w).reshape((h, w)).astype(np.uint8)


    # iterate through all pixels close to road plane
    for i in range(len(roadXs)):

        # reverse earlier projection and get px and py
        px = (roadXs[i] * focalLengthPX / roadZs[i]) + centreW
        py = centreH - (roadYs[i] * focalLengthPX / roadZs[i])

        # draw dot for each pixel deemed close to the road on blank image
        # plot large(ish) circle to blend dots into one shape (ideally)
        scatterRoad = cv2.circle(black, (int(px), int(py)), 7, (255), -1)


    scatterRoad = getROI(scatterRoad)

    # capture a contour of these scattered points
    _, threshold = cv2.threshold(scatterRoad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, contours, _ = cv2.findContours(threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    # return biggest contour
    biggestCont = max(contours, key = cv2.contourArea)
    return biggestCont


# Return the coords of each non-black pixel of the disparity image
# get only a proportion of points to reduce computation
def get3DPoints(disparity, proportion, removeOutliers):

    newImg = disparity.copy()
    newImg = convertGray(newImg)

    # Check @proportion is valid
    if (0 <= proportion) & (proportion <= 1) & isinstance(proportion, float):
        pass
    else:
        raise ValueError('Proportion is not a float or between 0 and 1')


    # e.g: if proportion is 0.2, get every 5th point
    every = int (1 / proportion)

    # check for whether or not to get points
    count = 0


    focalLengthPX = 399.9745178222656
    stereoBaseline = 0.2090607502
    steBasFocLen = focalLengthPX * stereoBaseline
    centreH = 262.0
    centreW = 474.5
    #Zmax = (focalLengthPX * stereoBaseline) / 2


    # Setup arrays to contains coords for each dimension
    Xs, Ys, Zs = np.array([]), np.array([]), np.array([])

    # Iterate over image
    h, w = np.shape(newImg)
    for py in range (h):
        for px in range (w):

            # check for every 5th coord
            if count == every:

                count = 0

                # if we have a valid non-zero disparity
                if newImg[py][px] > 0:

                    # calculate corresponding 3D point
                    Z = steBasFocLen / newImg[py][px]
                    X = (px - centreW) * Z / focalLengthPX
                    Y = (centreH - py) * Z / focalLengthPX      # invert y axis to follow normal cartesian coords (up is positive)

                    Xs = np.append(Xs, X)
                    Ys = np.append(Ys, Y)
                    Zs = np.append(Zs, Z)

            count += 1

    # now have coords for a proportion of points in disparity

    if removeOutliers == True:
        # For sanity if visualizing the scatter plot
        # and more importantly to ignore background ojects
        # Throws away anything more than two standard devs above the mean
        # ie. far away objects

        i = 0
        thresh = np.mean(Zs) + 2 * np.std(Zs)
        while i < len(Zs):
            if Zs[i] > thresh:
                Xs = np.delete(Xs, i)
                Ys = np.delete(Ys, i)
                Zs = np.delete(Zs, i)
            else:
                i += 1

    return Xs, Ys, Zs


# Calculate equn of plane for road based on 3D coords of a frame using ransac fitting algorithm
def getRoadPlaneEqu(Xs, Ys, Zs):

    # Check coord arrays are all same length
    if len(Xs) == len(Ys) == len(Zs):
        pass
    else:
        raise ValueError('Coord arrays are not same length :/')


    # reshape
    length = len(Xs)
    Xs, Ys, Zs = Xs.reshape((length, 1)), Ys.reshape((length, 1)), Zs.reshape((length, 1))

    # take road plane to be in XZ plane where X is across the bonnet, Y is up/down, Z is direction of motion

    # get RANSAC equ for line of X against Y
    # RANSAC class defined at bottom of this script
    # ie. across the bonnet of the car vs vertical direction
    # cross section of the road facing the direction of travel
    XYransac = RANSAC()
    XYransac.fit(Xs, Ys)


    # get RANSAC equ for line of Z against Y
    # ie. direction of motion vs vertical direction
    # cross section of the road and car would move in horizontal direction
    ZYransac = RANSAC()
    ZYransac.fit(Zs, Ys)

    # now I have a cross section of the plane of the road from X, Z viewpoints

    # Get some points to make a plane from these lines:
    # X = {1, 2}, Y = {corresponding RANSAC points for line in X-Y}, Z = 0
    point1XY = np.array([1, XYransac.predict(1), 0])
    point2XY = np.array([2, XYransac.predict(2), 0])

    # X = 0, Y = {corresponding RANSAC points for the Z-Y line}, Z = {1, 2}
    point1ZY = np.array([0, ZYransac.predict(1), 1])
    point2ZY = np.array([0, ZYransac.predict(2), 2])

    # normal from vectors of each RANSAC line
    normal = np.cross(point2ZY - point1ZY, point2XY - point1XY)

    # normalise
    magnitude = math.sqrt(  np.sum( [   normal[i]**2 for i in range (len(normal))   ] ) )
    normal *= 1.0 / magnitude

    # plane constant d for ax + by + cz = d     dot prod between normal and a point on plane
    dotProduct = np.sum([normal[i] * point2ZY[i] for i in range (len(normal))])

    # constants for plane ie. np.array([a, b, c, d])
    plane = np.append(normal, dotProduct)

    return plane


# Returns percent of pixels in the disparity which are not a black
# used for measuring the quality of the disparity match
# not implemented in main script at submission
def percentMatched(img):
    newImg = img.copy()
    newImg = convertGray(newImg)

    h, w = np.shape(newImg)
    nonBlack = len(np.nonzero(newImg)[1])   # number of non zero (non black) pixels

    percentMatched = 100 * round(float(nonBlack) / (h * w), 4)
    return percentMatched


# Returns gray image if input is not gray
def convertGray(img):
    newImg = img.copy()

    # If pixel value is an int (not triplet) then image is gray
    if isinstance(newImg[0][0], np.uint8):
        return newImg
    else:
        return cv2.cvtColor(newImg, cv2.COLOR_BGR2GRAY)


# I implemented this class based off the RANSAC algorithm displayed in wikipedia
# allows fitting a trendline using a random sample, the prediction using said trendline
class RANSAC:

    # instantiate new instance of class, setup some params used for fitting
    def __init__(self):

        self.coefs = np.array([])       # coefs for a linear model of regression i.e. m, c for y = mx + c - the fitted trendline
        self.noSampleData = None        # number of data points sampled to fit a model, set based on size of dataset
        self.maxIts = 10                # maximum number of iterations allowed in the algorithm
        self.fitThresh = 0.05           # threshold value to determine when a data point fits a model (5cm, is less than the kirb height)
        self.newModelThresh = None      # number of close data points required to assert that a model fits well to data, set based on size of dataset


    # return predicted value of input based on the fitted trendline
    def predict(self, X):

        if self.coefs != np.array([]):
            # return y = m * x + c
            return self.coefs[0] * X + self.coefs[1]

        else:
            raise ValueError('RANSAC objects has not been fitted!')


    # find a trendline using RANSAC algorith to fit the X and y data
    def fit(self, X, y):

        if len(X) != len(y):
            raise ValueError('X and y datasets are not same length!')

        self.noSampleData = int(len(X) / 20)        # set sample size to 20th of dataset
        self.newModelThresh = int(len(X) / 3)       # assert that 33% of data needs to be within 5cm of model to readjust the model before submitting


        its = 0                     # No of current iteration
        bestFit = None              # coefs of best iteration
        bestErr = 0                 # Error from best current fit

        while its < self.maxIts:


            # SELECTING SAMPLE DATA
            indexes = np.arange(len(X))                                                     # All indexes of X data
            sampleIndexes = np.array([random.sample(range(0, len(X)), self.noSampleData)])  # Selects noSampleData random indexes for X data

            otherIndexes = np.array( [i for i in indexes if i not in sampleIndexes] )       # Indexes of X data not in selected sampleIndexes

            sampleX = np.hstack(np.hstack(    [X[i] for i in sampleIndexes]   ))            # Select the sample data from the random indexes
            sampleY = np.hstack(np.hstack(    [y[i] for i in sampleIndexes]   ))


            # FITTING SAMPLE DATA
            currentCoefs = np.polyfit(
                        np.transpose(sampleX), np.transpose(sampleY), 1)        # fit linear trendline based on sample data - degree is 1: take roads to be flat planes
            yPred = lambda x: currentCoefs[0] * x + currentCoefs[1]             # function to predict output from X, based on found model (m*x + c)


            # FINDING OTHER INLIERS
            # for each data not in selected sample,
            # count as inlier if outcome is within error of predicted outcome
            otherXInliers = np.array([X[i] for i in otherIndexes if y[i] - yPred(X[i]) < self.fitThresh])
            otherYInliers = np.array([y[i] for i in otherIndexes if y[i] - yPred(X[i]) < self.fitThresh])


            # REFIT MODEL WITH ALL INLIER DATA
            if len(otherXInliers) >= self.newModelThresh:                       # implying found model is good

                allXInliers = np.append(sampleX, otherXInliers)
                allYInliers = np.append(sampleY, otherYInliers)

                betterCoefs = np.polyfit(
                    np.transpose(allXInliers), np.transpose(allYInliers), 1)    # model parameters fitted to all inlier points

                yBetterPred = lambda x: betterCoefs[0] * x + betterCoefs[1]


                # CHECK ACCURACY OF REFITED MODEL AND SUBMIT IF MORE ACCURATE
                r = np.corrcoef(
                        np.hstack(X), yBetterPred(np.hstack(X)))[0, 1]          # statistical value relating to closeness of data to trendline
                rSquared = r ** 2                                               # closer to 1 is more accurate

                if rSquared > bestErr:                                          # if more accurate than best yet, resubmit best model
                    bestFit = betterCoefs
                    bestErr = rSquared

            its += 1

        self.coefs = bestFit        # return best fit to RANSAC object



''' Main flow '''


#matches = np.array([])     # for containing percentages of disparity match over imgs


# Setup the disparity stereo processor
maxDisparity = 16 * 3
dispNoiseFilter = 5
stereoProcessor = cv2.StereoSGBM_create(
    minDisparity = 0, numDisparities = maxDisparity, blockSize = 21
)


# Iterate over dataset
for leftImg in leftFileList:


    # Collect image dirs
    rightImg = leftImg.replace("_L", "_R")
    pathLeftImg = os.path.join(leftPath, leftImg)
    pathRightImg = os.path.join(rightPath, rightImg)

    if (".png" in leftImg) and (os.path.isfile(pathRightImg)):


        # incase of image loading problems
        try:

            # Load and process the images for disparity matching
            imgL = cv2.imread(pathLeftImg, cv2.IMREAD_COLOR)
            imgR = cv2.imread(pathRightImg, cv2.IMREAD_COLOR)
            processedL = preprocessImg(imgL)
            processedR = preprocessImg(imgR)


            # Compute disparity
            disparity = stereoProcessor.compute(processedL, processedR)
            cv2.filterSpeckles(disparity, 0, 4000, maxDisparity - dispNoiseFilter)

            _, disparity = cv2.threshold(disparity, 0, maxDisparity * 16, cv2.THRESH_TOZERO)
            disparity_scaled = (disparity / 16.).astype(np.uint8)


            # Calculate the percentage of the disparity which has been matched from left and right
            # Allows an evaluation of the preprocessing methods
            #percentMatched = percentMatched(disparityShow)
            #matches = np.append(matches, percentMatched)
            #print ("Mean match rate of dataset: ", np.mean(matches))


            # output specified in Assignment
            print (leftImg)

            Xs, Ys, Zs = get3DPoints(getROI(disparity_scaled), 0.1, True)       # get 10% (0.1) of all coords of points in ROI, remove far away points
            roadPlane = getRoadPlaneEqu(Xs, Ys, Zs)                             # get eqn of plane using ransac from coords

            formattedNormal = '(' + str(list(roadPlane[:-1]))[1:-1] + ')'
            print (rightImg, ': road surface normal', formattedNormal)

            roadXs, roadYs, roadZs = closeToRoad(Xs, Ys, Zs, roadPlane, 0.1)    # get coords of points within 0.1 meters of plane height

            h, w = np.shape(processedL)
            roadContour = findContour(roadXs, roadYs, roadZs, h, w)             # find biggest contour of points close to plane
            hull = cv2.convexHull(roadContour)                                  # convert contour to polygon shape - removes details of raised objects
            cv2.drawContours(imgL, [hull], -1, (0, 0, 255), 6)                  # plot hull

            cv2.imshow('imgL', imgL)
            print ()


            # pause
            key = cv2.waitKey(40)


        except:
            print (leftImg)
            print (rightImg, ': road surface normal (0, 0, 0)')
            print ()


cv2.destroyAllWindows()
