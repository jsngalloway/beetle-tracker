import os
import sys
import numpy as np
import json
import math
import cv2
import random
from shutil import copyfile
import json
import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
# import matplotlib.pyplot as plt
# from matplotlib.patches import Circle

class Video:
    SESSION_NAME = 'script'
    START_TIME = 20000 #20 seconds, the time we initially jump to
    BRACKET_BUFFER = 10
    WINDOW_NAME = 'window'

    def __init__(self):
        print("made empty video")
        
    def initialize(self, p):
        self.path = p
        #TODO this is arbitrary
        self.proximity_range = 45
        print("THIS IS NOW IN THE NEW INITLIAZE FNCTION {}".format(self.path))
        self.vidcap = cv2.VideoCapture(self.path)
        self.videoFile = os.path.basename(self.path)

        self.length = int(self.vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.time = Video.START_TIME
        self.vidcap.set(cv2.CAP_PROP_POS_MSEC,self.time)
        success,self.image = self.vidcap.read()
        self.height, self.width = self.image.shape[:2]

        self.bracketROI = None
        self.basicArenaROI = None
        self.arenaROIstr = None
        self.black = None
        self.startFrame = None
        self.videoEndFrame = None
        # ------------------- End Init Section ---------------------

        if not success:
            print('An unknown error has occurred processing the video, contact Jesse.')
            print(self.path)



    def getBracketROIWrapper(self):
        self.bracketROI = self.getBracketROI(self.image, Video.WINDOW_NAME)
    def getArenaROIWrapper(self):
        (self.basicArenaROI, self.arenaROIstr) = self.getArenaROI2(self.image, Video.WINDOW_NAME)
    def getBeetleSelectWrapper(self):
        self.black = self.beetleSelect(self.image, Video.WINDOW_NAME)
    def getFirstFrameWrapper(self):
        self.startFrame = self.getFirstFrame(Video.WINDOW_NAME)
        self.videoEndFrame = self.length + self.startFrame
    def finishAndSave(self):
        cv2.destroyAllWindows()
        print("Processing Complete.\n\tStart Frame: {}/{}\n\tArena: {}\n\tBracket (x,y,w,h): {}".format(self.startFrame, self.length, self.arenaROIstr, self.bracketROI))
        self.savePostProcData()

    def getTrackingCommand(self):
            #TODO does video end at the end?
            cmd = "idtrackerai terminal_mode --_video \"{}\" --_session {} --_intensity [0,162] --_area [150,60000] --_range [{},{}] --_nblobs 2 --_roi \"{}\" --exec track_video".format(self.videoFile, Video.SESSION_NAME, self.startFrame, self.length, self.arenaROIstr)
            return cmd

    def beetleSelect(self, img, windowName):
        print("Click to select black beetle. Press any key to continue.")
        class CoordinateStore:
            def __init__(self, image):
                self.black = []
                self.old_image = np.copy(image)
                self.image = image

            def select_point(self,event,x,y,flags,param):
                if event == cv2.EVENT_LBUTTONDOWN:
                    if not self.black == []:
                        self.black = []
                        cv2.imshow(windowName, self.old_image)
                        self.image = np.copy(self.old_image)
                    self.black = [x,y]
                    self.image = cv2.circle(self.image, (x,y), 13, (0, 0, 0) , 2)
                    cv2.imshow(windowName, self.image)

            def getBlack(self):
                return self.black

        coordStore = CoordinateStore(img)

        cv2.imshow(windowName, img)
        cv2.setMouseCallback(windowName, coordStore.select_point)
        cv2.waitKey(0)
        cv2.setMouseCallback(windowName, lambda *args : None)
        retVal = coordStore.getBlack()
        return retVal

    def getArenaROI2(self, img, windowName):
        def contourToString(contour):
            point_lst = ["[["]
            first = True
            for point in contour:
                if not first:
                    point_str = ",({},{})".format(point[0][0],point[0][1])
                else:
                    point_str = "({},{})".format(point[0][0],point[0][1])
                    first = False
                point_lst.append(point_str)
            point_lst.append("]]")
            return ''.join(point_lst)

        class MaskManager:
            def __init__(self, img):
                self.mask = np.zeros(img.shape[:2],np.uint8)
                self.updated = False

            def draw_circle(self,event,x,y,flags,param):
                if event == cv2.EVENT_LBUTTONUP:
                    print("Marked areas as foreground, press enter to regenerate")
                    # cv2.circle(img,(x,y),8,(255,0,0),-1)
                    for i in range(y-10, y+10):
                        for j in range(x-10, x+10):
                            self.mask[i][j] = cv2.GC_FGD
                            # cv2.circle(disp_img,(j,i),1,(255,0,0),-1)
                    self.updated = True
                if event == cv2.EVENT_RBUTTONUP:
                    print("Marked areas as background, press enter to regenerate")
                    # cv2.circle(img,(x,y),8,(0,255,0),-1)
                    for i in range(y-10, y+10):
                        for j in range(x-10, x+10):
                            self.mask[i][j] = cv2.GC_BGD
                            # cv2.circle(disp_img,(j,i),1,(255,0,255),-1)
                    self.updated = True

            def resetUpdateFlag(self):
                self.updated = False
            def getUpdateFlag(self):
                return self.updated

            def getMask(self):
                return self.mask
            def setMask(self, nMask):
                self.mask = nMask

        mm = MaskManager(img)
        bgdModel = np.zeros((1,65),np.float64)
        fgdModel = np.zeros((1,65),np.float64)
        rect = cv2.selectROI(windowName, img)
        cv2.namedWindow(windowName)

       
        cv2.setMouseCallback(windowName,mm.draw_circle)

        msk, bgdModel, fgdModel = cv2.grabCut(img,mm.getMask(),rect,bgdModel,fgdModel,3,cv2.GC_INIT_WITH_RECT)
        mm.setMask(msk)
        contours = None
        while(1):

            # If mask==2 or mask== 1, mask2 get 0, other wise it gets 1 as 'uint8' type.
            mask2 = np.where((mm.getMask()==2)|(mm.getMask()==0),0,1).astype('uint8')
            img_cut = img*mask2[:,:,np.newaxis]

            imgray = cv2.cvtColor(img_cut, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(imgray, 127, 255, 0)
            ### Something funny happens with the next line depending on the version of opencv?
            _, contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            # contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            largest = 0
            for i in range(len(contours)):
                if cv2.contourArea(contours[i]) > cv2.contourArea(contours[largest]):
                    largest = i


            disp_img = np.copy(img_cut)
            cv2.drawContours(disp_img, contours, largest, (0,255,0), 2)
            cv2.imshow(windowName, disp_img)
            

            # wait for the user to hit enter to regen the model
            if cv2.waitKey(0):
                if not mm.getUpdateFlag():
                    break
                else:
                    mm.resetUpdateFlag()
                    print("Regenerating model")
                    msk, bgdModel, fgdModel = cv2.grabCut(img,mm.getMask(),None,bgdModel,fgdModel,3,cv2.GC_INIT_WITH_MASK)
                    mm.setMask(msk)
                    print("Done.")

        cv2.setMouseCallback(windowName, lambda *args : None)
        # cv2.destroyAllWindows()
        return(rect, contourToString(contours[largest]))

    def getArenaROI(self):
        print('Select the Beetle Arena....')
        r = cv2.selectROI(self.image)
        while (r == (0, 0, 0, 0)):
            self.time = self.time + 10000
            self.vidcap.set(cv2.CAP_PROP_POS_MSEC,self.time)
            success,self.image = self.vidcap.read()
            if success:
                print('Select the Beetle Arena...')
                r = cv2.selectROI(self.image)
            else:
                print('Error loading video frame.')
        #ROI is x,y,w,h
        a = (r[0], r[1])
        b = (r[0]+r[2], r[1])
        c = (r[0]+r[2], r[1]+r[3])
        d = (r[0], r[1]+r[3])

        ROIstr = "[[{},{},{},{}]]".format(a, b, c, d)
        print("Arena ROI confirmed.\n")
        return (r, ROIstr)

    def getBracketROI(self, img, windowName):
        print('Select the bracket....')
        bracket = cv2.selectROI("Select the bracket", img)
        cv2.destroyWindow("Select the bracket")
        print("Bracket ROI confirmed.\n")
        return bracket

    def getFirstFrame(self, windowName):

        def getBrightness(self, r, img):
            def getRandX(roi):
                return random.randint(r[0],r[0]+r[2])
            def getRandY(roi):
                return random.randint(r[1],r[1]+r[3])
            sum = 0
            for i in range (0, 500):
                x = getRandX(r)
                y = getRandY(r)
                # print("testing ({},{}) -> {}".format(x, y,image[x,y,0]))
                sum = sum + img[x,y,0]
            return sum/500

        self.vidcap.set(cv2.CAP_PROP_POS_FRAMES,0) #move to frame 0
        success, img = self.vidcap.read()
        strt = 0
        if success:
            strt = getBrightness(self, self.basicArenaROI, img)
        else:
            print("Error autodetecting brightness")

        def onChange(trackbarValue):
            print("trackbar change")
            self.vidcap.set(cv2.CAP_PROP_POS_FRAMES,trackbarValue)
            success,img = self.vidcap.read()
            cv2.imshow(windowName, img)
        
        # cv2.namedWindow(windowName)
        cv2.createTrackbar( 'start', windowName, 0, 600, onChange )
        print('Auto-detecting first frame...')
        for i in range(10, self.length, 5):
            self.vidcap.set(cv2.CAP_PROP_POS_FRAMES,i)
            success,image = self.vidcap.read()

            cv2.setTrackbarPos('start',windowName,i)
            # cv2.imshow(windowName, image)

            cfrm = getBrightness(self, self.basicArenaROI, image)
            if (cfrm > strt*1.5):
                # print("Frame {}, brightness: {}".format(i, cfrm))
                # self.vidcap.set(cv2.CAP_PROP_POS_FRAMES,i)
                # cv2.setTrackbarPos('start',windowName,i)
                print("Adjust or use ENTER to confirm Selection...")
                cv2.waitKey()
                startFrame = cv2.getTrackbarPos('start',windowName)
                print('First frame confirmed.')
                return startFrame

    def savePostProcData(self):
        video_id = os.path.splitext(os.path.basename(self.path))[0]
        target_dir = os.path.join(os.getcwd(), video_id)
        try:
            os.mkdir(target_dir)
        except OSError:
            print ("Creation of the directory %s failed, maybe it already exists..." % target_dir)
        else:
            print ("Successfully created %s folder" % target_dir)

        f = open(os.path.join(target_dir, (video_id + ".json")),"w+")
        class JsonOut:
            def __init__(self, cmd, location_black, startFrame, videoEndFrame, proximity_range, bracketROI, videoFile):
                self.cmd = cmd
                self.location_black = location_black
                self.startFrame = startFrame
                self.videoEndFrame = videoEndFrame
                self.proximity_range = proximity_range
                self.bracketROI = bracketROI
                self.video = videoFile

        jOut = JsonOut(cmd = self.getTrackingCommand(), location_black=self.black, startFrame=self.startFrame, videoEndFrame=self.videoEndFrame, proximity_range=self.proximity_range, bracketROI=self.bracketROI, videoFile=self.videoFile)
        jsonStr = json.dumps(jOut, indent=4, default=lambda o: o.__dict__)
        f.write(jsonStr)
        f.close()

        #copy the original video file into the new directory
        copyfile(self.path, os.path.join(target_dir, os.path.basename(self.path)))
        print("Successfully generated tracking files for {} at {}".format(video_id, target_dir))

if __name__ == '__main__' :
    # v = Video()
    # v.savePostProcData()
    print("Run the processing script on GPU cluster:\npython postprocessing.py {}".format(v.path))