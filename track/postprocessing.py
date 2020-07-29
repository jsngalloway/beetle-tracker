import os
import sys
import numpy as np
import json
import math
import cv2
import random
import csv
import string
import datetime

class Job:
    SESSION_NAME = 'script'
    BRACKET_BUFFER = 10
    WINDOW_NAME = 'window'
    VIDEO_OUTPUT_FPS = 10
    SECONDS_PER_FRAME = 4

    def __init__(self, data):
        self.path = data
        data_file = open("{}".format(self.path), "r")
        data = json.load(data_file)

        rawCmd = data['cmd']
        rawCmd.replace("\\\"", "\"")
        self.cmd = data['cmd']
        self.black = data['location_black']
        self.startFrame = data['startFrame']
        self.videoEndFrame = data['videoEndFrame']
        self.proximity_range = data['proximity_range']
        self.bracketROI = data['bracketROI']
        self.videoFile = data['video']

        data_file.close()

        # lines = data_file.readlines()
        # self.cmd = lines[0]
        # self.black = eval(lines[1])
        # self.startFrame = int(lines[2])
        # self.videoEndFrame = int(lines[3])
        # self.proximity_range = int(lines[4])
        # self.bracketROI = eval(lines[5])
        # print(self.cmd)
        # print(self.black)
    def getDirectory(self):
        return os.path.dirname(self.path)

    def postprocess(self):
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
        def interpolate(trajectories):
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

            return linear_interpolation(prox_interpolation(trajectories))
            # trajectories = linear_interpolation(trajectories)

        def verify_and_swap_black_white(trajectories):
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
            incident_list = []
            class Incident:
                def __init__(self,start_frame,white_avg_vel_in,black_avg_vel_in):
                    self.start_frame = start_frame
                    self.start_time_video = datetime.timedelta(seconds=start_frame/Job.VIDEO_OUTPUT_FPS)
                    self.start_time_trial = datetime.timedelta(seconds=start_frame*Job.SECONDS_PER_FRAME)
                    if(black_avg_vel_in > white_avg_vel_in):
                        self.initiator = "Black"
                        self.initiator_confidence = black_avg_vel_in - white_avg_vel_in
                    else:
                        self.initiator = "White"
                        self.initiator_confidence = white_avg_vel_in - black_avg_vel_in
                    self.end_time_video = None
                    self.end_time_trial = None
                    self.duration_video = None
                    self.duration_trial = None
                def endIncident(self, end_frame, white_avg_vel_out, black_avg_vel_out):
                    self.end_frame = end_frame
                    self.end_time_video = datetime.timedelta(seconds=self.end_frame/Job.VIDEO_OUTPUT_FPS)
                    self.end_time_trial = datetime.timedelta(seconds=self.end_frame*Job.SECONDS_PER_FRAME)
                    self.duration_video = datetime.timedelta(seconds=(self.end_frame - self.start_frame)/Job.VIDEO_OUTPUT_FPS)
                    self.duration_trial = datetime.timedelta(seconds=(self.end_frame - self.start_frame)*Job.SECONDS_PER_FRAME)
                def prettyPrint(self):
                        print('\tVideo start time: {}'.format(str(self.start_time_video)))
                        print('\tTrial start time: {}'.format(str(self.start_time_trial)))
                        print('\tInitiator: {}'.format(self.initiator))
                        print('\t\tConfidence: {}'.format(round(self.initiator_confidence,1)))
                        if(self.end_frame):
                            print('\tVideo end time: {}'.format(str(self.end_time_video)))
                            print('\tTrial end time: {}'.format(str(self.end_time_trial)))
                            print('\tDuration video: {}'.format(str(self.duration_video)))
                            print('\tDuration trial: {}'.format(str(self.duration_trial)))
                        else:
                            print("\tERROR: interaction has no end.")

            def calculateAvgVelInRange(self, i, trajectories):

            for i in range(PROXIMITY_FRAME_RANGE, len(proximity)-PROXIMITY_FRAME_RANGE):
                distance = dist(getBlack(trajectories[i]), getWhite(trajectories[i]))
                proximity[i] = distance
                if (distance < self.proximity_range):
                    proximity[i] = distance
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
                        this_incident = Incident(i, white_avg_vel, black_avg_vel)
                        incident_list.append(this_incident)
                        in_prox_last = True
                else:
                    if in_prox_last:
                        #The interaction just ended

                        incident_list[-1].endIncident(i, ???????????????)
                        print('Contact Incident {}:'.format(len(incident_list)))
                        incident_list[-1].prettyPrint()
                    in_prox_last = False
            return proximity
        
        def make_video(trajectories, proximities, video_name):
            vidcap = cv2.VideoCapture(self.videoFile)
            success,image = vidcap.read()
            height, width = image.shape[:2]
            out = cv2.VideoWriter(video_name,cv2.VideoWriter_fourcc('M','J','P','G'), Job.VIDEO_OUTPUT_FPS, (width,height))
            vidcap.set(cv2.CAP_PROP_POS_FRAMES,self.startFrame)
            # index = 0
            for index in range(len(trajectories)):
            # while(self.vidcap.isOpened()):
                success, frame = vidcap.read()
                if success:
                    # Bracket Box
                    frame = cv2.rectangle(frame, (self.bracketROI[0]-Job.BRACKET_BUFFER,self.bracketROI[1]-Job.BRACKET_BUFFER), (self.bracketROI[0]+self.bracketROI[2]+Job.BRACKET_BUFFER,self.bracketROI[1]+self.bracketROI[3]+Job.BRACKET_BUFFER), (100, 255, 0), 2)


                    frame = cv2.circle(frame, (int(getBlack(trajectories[index])[0]), int(getBlack(trajectories[index])[1])), 2, (0, 0, 0) , 3)
                    frame = cv2.circle(frame, (int(getWhite(trajectories[index])[0]), int(getWhite(trajectories[index])[1])), 2, (255, 255, 255) , 3)
                    
                    font                   = cv2.FONT_HERSHEY_SIMPLEX
                    bottomLeftCornerOfText = (40,40)
                    fontScale              = 0.66
                    fontColor              = (255,0,0)#BGR
                    lineType               = 2
                    cv2.putText(frame, "Dist: {}".format(round(proximities[index])), bottomLeftCornerOfText, font, fontScale, fontColor, lineType)

                    cv2.putText(frame, "On Bracket: {} {}".format("Black" if inBox(getBlack(trajectories[index]), self.bracketROI, Job.BRACKET_BUFFER) else "","White" if inBox(getWhite(trajectories[index]), self.bracketROI, Job.BRACKET_BUFFER) else ""), (40,70), font, fontScale, fontColor, lineType)
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
            vidcap.release()
            out.release()

            # Closes all the frames
            cv2.destroyAllWindows() 

        def save_csv(trajectories, proximities, file_name):
            with open(file_name, mode='w') as csv_file:
                data_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
                data_writer.writerow(['seconds','Black x', 'Black y', 'White x', 'White y', 'dist', 'proximity', 'Black on bracket', 'White on bracket'])
                time = 0
                for index in range(len(trajectories)):
                    time  = time + 4
                    data_writer.writerow([time, getBlack(trajectories[index])[0], getBlack(trajectories[index])[1], getWhite(trajectories[index])[0], getWhite(trajectories[index])[1], proximities[index], inBox(getBlack(trajectories[index]), self.bracketROI, Job.BRACKET_BUFFER), inBox(getWhite(trajectories[index]), self.bracketROI, Job.BRACKET_BUFFER)])

        #Load in trajectories 'without gaps'
        self.trajectories_wo_gaps_path = 'session_' + Job.SESSION_NAME + '\\trajectories_wo_gaps\\trajectories_wo_gaps.npy'
        # self.trajectories_wo_gaps_path = (os.path.dirname(os.path.realpath(self.path))) + '\\session_' + Job.SESSION_NAME + '\\trajectories_wo_gaps\\trajectories_wo_gaps.npy'
        trajectories_dict = np.load(self.trajectories_wo_gaps_path, allow_pickle=True).item()
        all_trajectories = trajectories_dict["trajectories"]

        #Crop the trajectories array down to the analysed part
        trajectories = all_trajectories[self.startFrame:self.videoEndFrame-1]
        print('---------- INTERPOLATION ----------')
        trajectories = verify_and_swap_black_white(interpolate(trajectories))
        # trajectories = verify_and_swap_black_white(trajectories)
        print('---------- PROXIMITY ANALYSIS ----------')
        proximities = proximity_detection(trajectories)
        save_csv(trajectories, proximities, "post_proc_output.csv")
        print('Data output saved to {}'.format("post_proc_output.csv"))
        # make_video(trajectories, proximities, "post_proc_output.avi")
        # print('Annotated video saved to {}'.format("post_proc_output.avi"))


if __name__ == '__main__' :
    j = Job(sys.argv[1])

    start_cwd = os.getcwd()
    os.chdir(j.getDirectory())

    # os.system(j.cmd)

    j.postprocess()

    os.chdir(start_cwd)
