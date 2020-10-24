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
from incident import Incident, ProximityIncident, TrialIndicent, StartIncident, BracketIncident
from InqscribeEvent import InqscribeEvent

class Job:
    SESSION_NAME = 'script'
    BRACKET_BUFFER = 10
    WINDOW_NAME = 'window'
    VIDEO_OUTPUT_FPS = 30
    SECONDS_PER_FRAME = 4

    def __init__(self, data):
        print(data)
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

    def getDirectory(self):
        return os.path.dirname(self.path)

    def frame2videoTime(self, frame):
        return datetime.timedelta(seconds=frame/Job.VIDEO_OUTPUT_FPS)

    def frame2trialTime(self, frame):
        return datetime.timedelta(seconds=frame*Job.SECONDS_PER_FRAME)

    def postprocess(self):
        
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
            # print("Black is {} away from id0.".format(black_to_id0))
            black_to_id1 = dist(self.black, trajectories[0][1])
            if (black_to_id0 > black_to_id1):
                #1 is black, 0 is white: they need to be swapped
                print("Beetle Ids swapped.")
                for i in range(len(trajectories)):
                    tmp = np.copy(trajectories[i][0])
                    trajectories[i][0] = trajectories[i][1]
                    trajectories[i][1] = tmp
            black_to_id0 = dist(self.black, trajectories[0][0])
            # print("Black is {} away from id0.".format(black_to_id0))
            return trajectories
        def find_all_incidents(trajectories):
            incidents = []
            def proximity_detection(trajectories):
                proximity = np.empty(len(trajectories), float)
                # velocity_to_avg_pos = np.empty((len(trajectories), 2), float)

                in_prox_last = False
                PROXIMITY_FRAME_RANGE = 10
                incident_list = []


                def calculateAvgVelInRange(i, trajectories, frame_range):
                    avg_pos = [(getBlack(trajectories[i])[0]+getWhite(trajectories[i])[0])/2, (getBlack(trajectories[i])[1]+getWhite(trajectories[i])[1])/2 ]
                    incident = trajectories[i-frame_range:i+frame_range]
                    black_avg_vel = 0
                    white_avg_vel = 0
                    for j in range(1, len(incident)):
                        black_avg_vel += dist(getBlack(incident[j-1]), avg_pos) - dist(getBlack(incident[j]), avg_pos)
                        white_avg_vel += dist(getWhite(incident[j-1]), avg_pos) - dist(getWhite(incident[j]), avg_pos)
                    black_avg_vel = black_avg_vel/(len(incident)-1)
                    white_avg_vel = white_avg_vel/(len(incident)-1)
                    return white_avg_vel, black_avg_vel

                for i in range(PROXIMITY_FRAME_RANGE, len(proximity)-PROXIMITY_FRAME_RANGE):
                    distance = dist(getBlack(trajectories[i]), getWhite(trajectories[i]))
                    proximity[i] = distance
                    if (distance < self.proximity_range):
                        proximity[i] = distance
                        if not in_prox_last:
                            white_avg_vel, black_avg_vel = calculateAvgVelInRange(i, trajectories, PROXIMITY_FRAME_RANGE)
                            this_incident = ProximityIncident(self.frame2videoTime(i), white_avg_vel, black_avg_vel)
                            incident_list.append(this_incident)
                            in_prox_last = True
                    else:
                        if in_prox_last:
                            #The interaction just ended
                            white_avg_vel, black_avg_vel = calculateAvgVelInRange(i, trajectories, PROXIMITY_FRAME_RANGE)
                            incident_list[-1].endIncident(self.frame2videoTime(i), white_avg_vel, black_avg_vel)
                        in_prox_last = False
                return proximity, incident_list
            def find_first_move(trajectories, beetleId):
                beetle_char = 'b'
                if beetleId == 1:
                    beetle_char = 'w'

                start = [trajectories[0][beetleId][0], trajectories[0][beetleId][1]]
                frame_count = 0
                for frame in trajectories:
                    if dist(start, frame[beetleId])  > 10:
                        return StartIncident(self.frame2videoTime(frame_count), beetle_char)
                    frame_count += 1
            def get_bracket_incidents(trajectories, beetleId):
                # create a history buffer (fill it with 0, 1 means on bracket)
                history = [False] * 5

                beetleChar = 'b'
                if beetleId == 1:
                    beetleChar = 'w'

                events = []
                frame_count = 0
                current_event = None
                for frame in trajectories:
                    on_bracket = inBox(frame[beetleId], self.bracketROI, Job.BRACKET_BUFFER)
                    # print(history)
                    if history.pop(): #remove the oldest element in the list, effectivly a queue
                        if True not in history:
                            if current_event:
                                current_event.endIncident(self.frame2videoTime(frame_count))
                                events.append(current_event)
                                current_event = None
                            else:
                                print('we got an issue here bud')
                    if (on_bracket and (current_event == None)):
                        current_event = BracketIncident(self.frame2videoTime(frame_count), beetleChar)
                    history.insert(0, on_bracket) #insert whether the beetle was on the bracket at the beginning of the list
                    frame_count += 1
                return events


            proximity_incidents = []
            proximities, proximity_incidents = proximity_detection(trajectories)
            print("Found: [{}] proximity incidents".format(len(proximity_incidents)))
            incidents.extend(proximity_incidents)

            trial_incidents = TrialIndicent(self.frame2videoTime(0), self.frame2videoTime(len(trajectories)))
            print("Found: [1] trial incidents")
            incidents.append(trial_incidents)

            first_move_incidents = []
            first_move_incidents.append(find_first_move(trajectories, 0))#beetle 0 is black
            first_move_incidents.append(find_first_move(trajectories, 1))#beetle 1 is white
            print("Found: [{}] first-move incidents".format(len(first_move_incidents)))
            incidents.extend(first_move_incidents)

            bracket_incidents = []
            bracket_incidents.extend(get_bracket_incidents(trajectories, 0))
            bracket_incidents.extend(get_bracket_incidents(trajectories, 1))
            print("Found: [{}] bracket incidents".format(len(bracket_incidents)))
            incidents.extend(bracket_incidents)
            return proximities, incidents

        def save_inqscribe(filename, events):
            events.insert(0, (InqscribeEvent(self.frame2videoTime(0), 'Note:' + filename, None)))
            f = open(filename + ".inqscr", mode="w")
            f.write("app=InqScribe\n")
            event_str = ""
            for event in events:
               event_str += (str(event) + "\\r")
            f.write("text={}".format(event_str))
            f.close()
    
        def make_video(trajectories, proximities, inqscribe_events: list, video_name):
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
                    
                    cv2.putText(frame, "{}".format(str(self.frame2videoTime(index)).split('.', 2)[0]), (40,100), font, fontScale, fontColor, lineType)
                    if len(inqscribe_events):
                        cv2.putText(frame, "Past:{}".format(inqscribe_events[0]), (40,130), font, fontScale, fontColor, lineType)
                    if len(inqscribe_events) > 1:
                        cv2.putText(frame, "Next:{}".format(inqscribe_events[1]), (40,160), font, fontScale, fontColor, lineType)
                        if (self.frame2videoTime(index+16) >= (inqscribe_events[1]).getTime()):
                            inqscribe_events.pop(0)
                    if len(inqscribe_events) > 2:
                        cv2.putText(frame, "     {}".format(inqscribe_events[2]), (40,190), font, fontScale, fontColor, lineType)


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
                data_writer.writerow(['seconds','Black x', 'Black y', 'White x', 'White y', 'In Proximity', 'Black on bracket', 'White on bracket'])
                time = 0
                for index in range(len(trajectories)):
                    time  = time + Job.SECONDS_PER_FRAME
                    data_writer.writerow([time,
                                        getBlack(trajectories[index])[0],
                                        getBlack(trajectories[index])[1],
                                        getWhite(trajectories[index])[0],
                                        getWhite(trajectories[index])[1],
                                        proximities[index],
                                        inBox(getBlack(trajectories[index]), self.bracketROI, Job.BRACKET_BUFFER),
                                        inBox(getWhite(trajectories[index]), self.bracketROI, Job.BRACKET_BUFFER)])

        #Load in trajectories 'without gaps'
        self.trajectories_wo_gaps_path = 'session_' + Job.SESSION_NAME + '\\trajectories_wo_gaps\\trajectories_wo_gaps.npy'
        trajectories_dict = np.load(self.trajectories_wo_gaps_path, allow_pickle=True).item()
        all_trajectories = trajectories_dict["trajectories"]

        #Crop the trajectories array down to the analysed part
        trajectories = all_trajectories[self.startFrame:self.videoEndFrame-1]
        print('---------- INTERPOLATION ----------')
        trajectories = verify_and_swap_black_white(interpolate(trajectories))

        print('---------- INCIDENT ANALYSIS ----------')
        proximities, incidents = find_all_incidents(trajectories)
        print("TOTAL: [{}] incidents found".format(len(incidents)))
        
        print('---------- DATA EXPORT ----------')
        save_csv(trajectories, proximities, "post_proc_output.csv")
        print('Position data saved to: {}'.format("post_proc_output.csv"))
        
        inqscribe_events = []
        for incident in incidents:
            for event in incident.inqscribe_events:
                inqscribe_events.append(event)
        inqscribe_events.sort()
        save_inqscribe(self.path, inqscribe_events)
        print('Inqscribe file saved to: {}'.format("post_proc_output.csv"))

        # for incident in incidents:
        #     incident.prettyPrint()

        # for event in inqscribe_events:
        #     print(event)    

        make_video(trajectories, proximities, inqscribe_events, "post_proc_output.avi")
        print('Annotated video saved to {}'.format("post_proc_output.avi"))
        

if __name__ == '__main__' :
    j = Job(sys.argv[1])

    start_cwd = os.getcwd()
    os.chdir(j.getDirectory())
    os.system(j.cmd)

    print("Tracking complete. Beginning analysis...")
    j.postprocess()

    os.chdir(start_cwd)
