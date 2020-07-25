import os
import sys
import numpy as np
import json
import math
import cv2
import random
# import matplotlib.pyplot as plt
# from matplotlib.patches import Circle


class Video:
    SESSION_NAME = 'script'
    START_TIME = 20000 #20 seconds, the time we initially jump to
    BRACKET_BUFFER = 10
    WINDOW_NAME = 'window'

    def __init__(self, p): 
        self.path = p
        #TODO this is arbitrary
        self.proximity_range = 45
        print("Loaded in video {}".format(self.path))
        self.vidcap = cv2.VideoCapture(self.path)
        self.length = int(self.vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.time = Video.START_TIME
        self.vidcap.set(cv2.CAP_PROP_POS_MSEC,self.time)
        success,self.image = self.vidcap.read()
        self.height, self.width = self.image.shape[:2]

        if success:
            cv2.namedWindow(Video.WINDOW_NAME)

            self.bracketROI = self.getBracketROI(self.image, Video.WINDOW_NAME)
            
            (self.basicArenaROI, self.arenaROIstr) = self.getArenaROI2(self.image, Video.WINDOW_NAME)
            
            self.black = self.beetleSelect(self.image, Video.WINDOW_NAME)

            self.startFrame = self.getFirstFrame(Video.WINDOW_NAME)
            self.videoEndFrame = self.length + self.startFrame

            
            cv2.destroyAllWindows()
            print("Processing Complete.\n\tStart Frame: {}/{}\n\tArena: {}\n\tBracket (x,y,w,h): {}".format(self.startFrame, self.length, self.arenaROIstr, self.bracketROI))
        else:
            print('Unable to load video, verify video path or video may be very short.')
            print(self.path)

    def getTrackingCommand(self):
            #TODO does video end at the end?
            cmd = "idtrackerai terminal_mode --_video \"{}\" --_session {} --_intensity [0,162] --_area [150,60000] --_range [{},{}] --_nblobs 2 --_roi \"{}\" --exec track_video".format(self.path, Video.SESSION_NAME, self.startFrame, self.length, self.arenaROIstr)
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
                    self.updated = True
                    print("FOREGROUND")
                    # cv2.circle(img,(x,y),8,(255,0,0),-1)
                    for i in range(y-10, y+10):
                        for j in range(x-10, x+10):
                            self.mask[i][j] = cv2.GC_FGD
                            # cv2.circle(disp_img,(j,i),1,(255,0,0),-1)
                if event == cv2.EVENT_RBUTTONUP:
                    self.updated = True
                    print("BACKGROUND")
                    # cv2.circle(img,(x,y),8,(0,255,0),-1)
                    for i in range(y-10, y+10):
                        for j in range(x-10, x+10):
                            self.mask[i][j] = cv2.GC_BGD
                            # cv2.circle(disp_img,(j,i),1,(255,0,255),-1)

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
            # _, contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            largest = 0
            for i in range(len(contours)):
                if cv2.contourArea(contours[i]) > cv2.contourArea(contours[largest]):
                    largest = i
            # print(contours[largest])


            disp_img = np.copy(img_cut)
            cv2.drawContours(disp_img, contours, largest, (0,255,0), 2)
            cv2.imshow(windowName, disp_img)
            
            if cv2.waitKey(0):
                if not mm.getUpdateFlag():
                    break
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
        bracket = cv2.selectROI(windowName, img)
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

    # def postprocess(self):
        print("Tracking complete. Beginning trajectory analysis...")
        def dist(pointA,pointB): 
            d = math.sqrt((pointB[0] - pointA[0])**2 + (pointB[1] - pointA[1])**2)  
            return d
        def getBlack(frame):
            return frame[0]
        def getWhite(frame):
            return frame[1]
        def inBox(point, ROI, buffer=0):
            xMin = ROI[0] - buffer
            yMin = ROI[1] - buffer
            xMax = ROI[0] + ROI[2] + buffer
            yMax = ROI[1] + ROI[3] + buffer
            return ((xMin < point[0] < xMax) and (yMin < point[1] < yMax))
        
        def linear_interpolation(trajectories):
            interp = 0
            for id in range(0, 2):
                for axis in range(0, 2):
                    for frame in range(len(trajectories)):
                        if(np.isnan(trajectories[frame][id][axis])):
                            lastGood = frame-1
                            newGood = lastGood
                            endFrame = frame
                            while(np.isnan(trajectories[frame][id][axis])):
                                if(frame == len(trajectories)-1):
                                    print("we reached the end of the traj with no new good value, using uniform interpolation. Accuracy will suffer")
                                    endFrame = frame
                                    break
                                frame += 1
                            newGood = frame
                            endFrame = frame

                            if (newGood == lastGood):
                                for cr in range (lastGood, endFrame):
                                    trajectories[cr][id][axis] = trajectories[lastGood][id][axis]
                                    interp += 1
                            else:
                                step = (trajectories[newGood][id][axis] - trajectories[lastGood][id][axis])/(endFrame - lastGood)
                                for cr in range (lastGood, endFrame):
                                    trajectories[cr][id][axis] = trajectories[cr-1][id][axis] + step
                                    interp += 1
            print("Linear Interpolation finished. Filled ({}) values".format(interp))
            print("\t{}/{} values linearly interpolated, {}%".format(interp, len(trajectories)*4, (interp/(len(trajectories)*4))*100))
            return trajectories

        def prox_interpolation(trajectories):
            failed_frames = 0
            def copyGoodValues(frame):
                success = True
                #We'll only check the x coord since idtracker ai won't record just one axis
                if (np.isnan((frame)[0][0]) and (not np.isnan(frame[1][0]))):
                    #id 0 is gone but 1 is there, copy 1's values to 0
                    frame[0] = frame[1]
                elif (np.isnan((frame)[1][0]) and (not np.isnan(frame[0][0]))):
                    frame[1] = frame[0]
                else:
                    success = False
                return (success, frame)

            interp = 0
            for frame in trajectories:
                if np.isnan(np.sum(frame)):
                    interp += 1
                    success, frame = copyGoodValues(frame)
                    if not success: failed_frames += 1
            print("Proximity Interpolation finished. Filled ({}) values. ({}) Failed frames.".format(interp*2, failed_frames))
            print("\t{}/{} values proximity interpolated, {}%".format(interp*2, len(trajectories)*4, (interp/(len(trajectories)*2))*100))
            return trajectories


        def black_white_id(trajectories):
            black_to_id0 = dist(self.black, trajectories[0][0])
            print("Black is {} away from id0.".format(black_to_id0))
            black_to_id1 = dist(self.black, trajectories[0][1])
            if (black_to_id0 > black_to_id1):
                #1 is black, 0 is white: they need to be swapped
                print("Swapping beetle ids...")
                for i in range(len(trajectories)):
                    tmp = np.copy(trajectories[i][0])
                    trajectories[i][0] = trajectories[i][1]
                    trajectories[i][1] = tmp
            black_to_id0 = dist(self.black, trajectories[0][0])
            print("Black is {} away from id0.".format(black_to_id0))
            return trajectories
        
        def proximity_detection(trajectories):

            proximity = np.empty(len(trajectories), float)
            # velocity_to_avg_pos = np.empty((len(trajectories), 2), float)

            count = 0
            in_prox_last = False
            PROXIMITY_FRAME_RANGE = 10
            for i in range(PROXIMITY_FRAME_RANGE, len(proximity)-PROXIMITY_FRAME_RANGE):
                distance = dist(getBlack(trajectories[i]), getWhite(trajectories[i]))
                proximity[i] = distance
                if ( distance < self.proximity_range):
                    proximity[i] = distance
                    count += 1
                    if not in_prox_last:
                        avg_pos = [(getBlack(trajectories[i])[0]+getWhite(trajectories[i])[0])/2, (getBlack(trajectories[i])[1]+getWhite(trajectories[i])[1])/2 ]
                        #The beetles have just entered proximity, who initiated?
                        incident = trajectories[i-PROXIMITY_FRAME_RANGE:i+PROXIMITY_FRAME_RANGE]
                        black_avg_vel = 0
                        white_avg_vel = 0
                        for j in range(1, len(incident)):
                            black_avg_vel += dist(getBlack(incident[j-1]), avg_pos) - dist(getBlack(incident[j]), avg_pos)
                            white_avg_vel += dist(getWhite(incident[j-1]), avg_pos) - dist(getWhite(incident[j]), avg_pos)
                        black_avg_vel = black_avg_vel/(len(incident)-1)
                        white_avg_vel = white_avg_vel/(len(incident)-1)
                        print("Contact initiated, beetles are: {} pixels apart".format(dist(getBlack(trajectories[i]), getWhite(trajectories[i]))))
                        if(black_avg_vel > white_avg_vel):
                            print("Frame: {} Black initiates contact".format(i+self.startFrame))
                        else:
                            print("White intiates contact")
                        in_prox_last = True
                else:
                    in_prox_last = False
            print("frames in prox: {}/{}".format(count, len(trajectories)))
            return proximity
        
        def make_video(trajectories, proximities, video_name):
            out = cv2.VideoWriter(video_name,cv2.VideoWriter_fourcc('M','J','P','G'), 10, (self.width,self.height))
            self.vidcap.set(cv2.CAP_PROP_POS_FRAMES,self.startFrame)
            # index = 0
            for index in range(len(trajectories)):
            # while(self.vidcap.isOpened()):
                success, frame = self.vidcap.read()
                if success:
                    # Bracket Box
                    frame = cv2.rectangle(frame, (self.bracketROI[0]-Video.BRACKET_BUFFER,self.bracketROI[1]-Video.BRACKET_BUFFER), (self.bracketROI[0]+self.bracketROI[2]+Video.BRACKET_BUFFER,self.bracketROI[1]+self.bracketROI[3]+Video.BRACKET_BUFFER), (100, 255, 0), 2)


                    frame = cv2.circle(frame, (int(getBlack(trajectories[index])[0]), int(getBlack(trajectories[index])[1])), 2, (0, 0, 0) , 3)
                    frame = cv2.circle(frame, (int(getWhite(trajectories[index])[0]), int(getWhite(trajectories[index])[1])), 2, (255, 255, 255) , 3)
                    
                    font                   = cv2.FONT_HERSHEY_SIMPLEX
                    bottomLeftCornerOfText = (40,40)
                    fontScale              = 0.66
                    fontColor              = (255,0,0)#BGR
                    lineType               = 2
                    cv2.putText(frame, "Dist: {}".format(round(proximities[index])), bottomLeftCornerOfText, font, fontScale, fontColor, lineType)

                    cv2.putText(frame, "On Bracket: {} {}".format("Black" if inBox(getBlack(trajectories[index]), self.bracketROI, Video.BRACKET_BUFFER) else "","White" if inBox(getWhite(trajectories[index]), self.bracketROI, Video.BRACKET_BUFFER) else ""), (40,70), font, fontScale, fontColor, lineType)
                    if(proximities[index] < self.proximity_range):
                        frame = cv2.circle(frame, (int(getBlack(trajectories[index])[0]), int(getBlack(trajectories[index])[1])), self.proximity_range, (0, 0, 255) , 2)
                    # add annotations to frame
                    out.write(frame)

                    # Display the resulting frame    
                    # cv2.imshow('frame',frame)
                else:
                    break
                # index += 1

            # When everything done, release the video capture and video write objects
            self.vidcap.release()
            out.release()

            # Closes all the frames
            cv2.destroyAllWindows() 

        #Load in trajectories 'without gaps'
        self.trajectories_wo_gaps_path = (os.path.dirname(os.path.realpath(self.path))) + '\\session_' + Video.SESSION_NAME + '\\trajectories_wo_gaps\\trajectories_wo_gaps.npy'
        trajectories_dict = np.load(self.trajectories_wo_gaps_path, allow_pickle=True).item()
        all_trajectories = trajectories_dict["trajectories"]

        #Crop the trajectories array down to the analysed part
        trajectories = all_trajectories[self.startFrame:self.length-1]
        
        trajectories = prox_interpolation(trajectories)
        trajectories = linear_interpolation(trajectories)
        trajectories = black_white_id(trajectories)

        proximities = proximity_detection(trajectories)
        make_video(trajectories, proximities, "post_proc_output.avi")

    def savePostProcData(self):
        f= open(self.path + ".txt","w+")
        toWrite = "{}\n{}\n{}\n{}\n{}\n{}".format(self.getTrackingCommand(), self.black, self.startFrame, self.videoEndFrame, self.proximity_range, self.bracketROI)
        f.write(toWrite)
        f.close()

if __name__ == '__main__' :
    v = Video(sys.argv[1])
    v.savePostProcData()
    print("Run the processing script on GPU cluster:\npython postprocessing.py {}".format(v.path))