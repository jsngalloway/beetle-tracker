import os
from pathlib import Path
import sys
from typing import Any, Optional, Tuple
import numpy as np
import json
import math
import cv2
import csv
import datetime
from tqdm import tqdm
from incident import (
    ProximityIncident,
    TrialIncident,
    StartIncident,
    BracketIncident,
)
from InqscribeEvent import InqscribeEvent
import logging

logger = logging.getLogger(__name__)


class Job:
    SESSION_NAME = "script"
    BRACKET_BUFFER = 10
    WINDOW_NAME = "window"
    VIDEO_OUTPUT_FPS = 30
    SECONDS_PER_FRAME = 4

    def __init__(self, data):
        logger.info(data)
        self.path = data
        data_file = open("{}".format(self.path), "r")
        data = json.load(data_file)

        rawCmd = data["cmd"]
        rawCmd.replace('\\"', '"')
        self.cmd = data["cmd"]
        self.black = data["location_black"]
        self.startFrame = data["startFrame"]
        self.videoEndFrame = data["videoEndFrame"]
        self.proximity_range = data["proximity_range"]
        self.bracketROI = data["bracketROI"]
        self.videoFile = data["video"]
        self.options = data["options"]
        data_file.close()

    def getDirectory(self):
        return os.path.dirname(self.path)

    def frame2videoTime(self, frame):
        return datetime.timedelta(seconds=frame / Job.VIDEO_OUTPUT_FPS)

    def frame2trialTime(self, frame):
        return datetime.timedelta(seconds=frame * Job.SECONDS_PER_FRAME)

    def postprocess(self):

        def dist(pointA, pointB):
            d = math.sqrt((pointB[0] - pointA[0]) ** 2 + (pointB[1] - pointA[1]) ** 2)
            return d

        def getBlack(frame) -> Tuple[float, float]:
            return frame[0]

        def getWhite(frame) -> Tuple[float, float]:
            return frame[1]

        def inBox(point, ROI, buffer=0):
            xMin = ROI[0] - buffer
            yMin = ROI[1] - buffer
            xMax = ROI[0] + ROI[2] + buffer
            yMax = ROI[1] + ROI[3] + buffer
            return (xMin < point[0] < xMax) and (yMin < point[1] < yMax)

        def interpolate(trajectories):
            def linear_interpolation(trajectories):
                interp = 0
                for id in range(0, 2):
                    for axis in range(0, 2):
                        for frame in range(len(trajectories)):
                            if np.isnan(trajectories[frame][id][axis]):
                                lastGood = frame - 1
                                newGood = lastGood
                                endFrame = frame
                                while np.isnan(trajectories[frame][id][axis]):
                                    if frame == len(trajectories) - 1:
                                        logger.warning(
                                            f"Failed to perform linear interpolation for beetle '{id}' on axis '{axis}'"
                                            f" starting at frame {lastGood}. Using uniform interpolation."
                                        )
                                        endFrame = frame
                                        break
                                    frame += 1
                                newGood = frame
                                endFrame = frame

                                if newGood == lastGood:
                                    for cr in range(lastGood, endFrame):
                                        trajectories[cr][id][axis] = trajectories[
                                            lastGood
                                        ][id][axis]
                                        interp += 1
                                else:
                                    step = (
                                        trajectories[newGood][id][axis]
                                        - trajectories[lastGood][id][axis]
                                    ) / (endFrame - lastGood)
                                    for cr in range(lastGood, endFrame):
                                        trajectories[cr][id][axis] = (
                                            trajectories[cr - 1][id][axis] + step
                                        )
                                        interp += 1
                logger.info(
                    "Linear Interpolation finished. Filled ({}) values".format(interp)
                )
                logger.info(
                    "\t{}/{} values linearly interpolated, {}%".format(
                        interp,
                        len(trajectories) * 4,
                        (interp / (len(trajectories) * 4)) * 100,
                    )
                )
                return trajectories

            def prox_interpolation(trajectories):
                failed_frames = 0

                def copyGoodValues(frame):
                    success = True
                    # We'll only check the x coord since idtracker ai won't record just one axis
                    if np.isnan((frame)[0][0]) and (not np.isnan(frame[1][0])):
                        # id 0 is gone but 1 is there, copy 1's values to 0
                        frame[0] = frame[1]
                    elif np.isnan((frame)[1][0]) and (not np.isnan(frame[0][0])):
                        frame[1] = frame[0]
                    else:
                        success = False
                    return (success, frame)

                interp = 0
                for frame in trajectories:
                    if np.isnan(np.sum(frame)):
                        interp += 1
                        success, frame = copyGoodValues(frame)
                        if not success:
                            failed_frames += 1
                logger.info(
                    "Proximity Interpolation finished. Filled ({}) values. ({}) Failed frames.".format(
                        interp * 2, failed_frames
                    )
                )
                logger.info(
                    "\t{}/{} values proximity interpolated, {}%".format(
                        interp * 2,
                        len(trajectories) * 4,
                        (interp / (len(trajectories) * 2)) * 100,
                    )
                )
                return trajectories

            return linear_interpolation(prox_interpolation(trajectories))
            # trajectories = linear_interpolation(trajectories)

        def verify_and_swap_black_white(trajectories: list):
            black_to_id0 = dist(self.black, trajectories[0][0])
            # logger.info("Black is {} away from id0.".format(black_to_id0))
            black_to_id1 = dist(self.black, trajectories[0][1])
            if black_to_id0 > black_to_id1:
                # 1 is black, 0 is white: they need to be swapped
                logger.info("Beetle Ids swapped.")
                for i in range(len(trajectories)):
                    tmp = np.copy(trajectories[i][0])
                    trajectories[i][0] = trajectories[i][1]
                    trajectories[i][1] = tmp
            black_to_id0 = dist(self.black, trajectories[0][0])
            # logger.info("Black is {} away from id0.".format(black_to_id0))
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
                    avg_pos = [
                        (getBlack(trajectories[i])[0] + getWhite(trajectories[i])[0])
                        / 2,
                        (getBlack(trajectories[i])[1] + getWhite(trajectories[i])[1])
                        / 2,
                    ]
                    incident = trajectories[i - frame_range : i + frame_range]
                    black_avg_vel = 0
                    white_avg_vel = 0
                    for j in range(1, len(incident)):
                        black_avg_vel += dist(
                            getBlack(incident[j - 1]), avg_pos
                        ) - dist(getBlack(incident[j]), avg_pos)
                        white_avg_vel += dist(
                            getWhite(incident[j - 1]), avg_pos
                        ) - dist(getWhite(incident[j]), avg_pos)
                    black_avg_vel = black_avg_vel / (len(incident) - 1)
                    white_avg_vel = white_avg_vel / (len(incident) - 1)
                    return white_avg_vel, black_avg_vel

                for i in range(len(proximity)):
                    distance = dist(
                        getBlack(trajectories[i]), getWhite(trajectories[i])
                    )
                    proximity[i] = distance
                    if (i > PROXIMITY_FRAME_RANGE) and (
                        i < len(proximity) - PROXIMITY_FRAME_RANGE
                    ):
                        if distance < self.proximity_range:
                            if not in_prox_last:
                                white_avg_vel, black_avg_vel = calculateAvgVelInRange(
                                    i, trajectories, PROXIMITY_FRAME_RANGE
                                )
                                this_incident = ProximityIncident(
                                    self.frame2videoTime(i),
                                    white_avg_vel,
                                    black_avg_vel,
                                )
                                incident_list.append(this_incident)
                                in_prox_last = True
                        else:
                            if in_prox_last:
                                # The interaction just ended
                                white_avg_vel, black_avg_vel = calculateAvgVelInRange(
                                    i, trajectories, PROXIMITY_FRAME_RANGE
                                )
                                incident_list[-1].endIncident(
                                    self.frame2videoTime(i),
                                    white_avg_vel,
                                    black_avg_vel,
                                )
                            in_prox_last = False

                return proximity, incident_list

            def find_first_move(trajectories, beetleId: int):
                # Trajectory is formatted [ [blackX, blackY], [whiteX, whiteY] ], ...

                beetle_char = "b"
                if beetleId == 1:
                    beetle_char = "w"

                # Find the first frame where the beetle's position is not a nan
                start_frame = None
                for idx, frame in enumerate(trajectories):
                    if not np.isnan(frame[beetleId][0]) and not np.isnan(
                        frame[beetleId][1]
                    ):
                        logger.debug(
                            f"Found first non-nan position for beetle '{beetle_char}' "
                            f"at index {idx}/{len(trajectories)}"
                        )
                        start_frame = idx
                        break

                if start_frame is None:
                    logger.error(
                        f"Beetle '{beetle_char}' has no position throughout the test!"
                        " Unable to determine first mover incident."
                    )
                    return None

                start_position = [
                    trajectories[start_frame][beetleId][0],
                    trajectories[start_frame][beetleId][1],
                ]
                frame_count = 0
                for frame in trajectories:
                    if dist(start_position, frame[beetleId]) > 10:
                        logger.debug(
                            f"Beetle '{beetle_char}' moved far enough to register as a mover at frame ({frame_count})"
                        )
                        return StartIncident(
                            self.frame2videoTime(frame_count), beetle_char
                        )
                    frame_count += 1
                logger.info(
                    f"WARNING! No first mover incident found for beetle: {beetleId}. Did this beetle move?"
                )

            def get_bracket_incidents(trajectories, beetleId):
                # create a history buffer (fill it with 0, 1 means on bracket)
                history = [False] * 5

                beetleChar = "b"
                if beetleId == 1:
                    beetleChar = "w"

                events = []
                frame_count = 0
                current_event = None
                for frame in trajectories:
                    on_bracket = inBox(
                        frame[beetleId], self.bracketROI, Job.BRACKET_BUFFER
                    )
                    # logger.info(history)
                    if (
                        history.pop()
                    ):  # remove the oldest element in the list, effectively a queue
                        if True not in history:
                            if current_event:
                                current_event.endIncident(
                                    self.frame2videoTime(frame_count)
                                )
                                events.append(current_event)
                                current_event = None
                            else:
                                logger.info("we got an issue here bud")
                    if on_bracket and (current_event is None):
                        current_event = BracketIncident(
                            self.frame2videoTime(frame_count), beetleChar
                        )
                    history.insert(
                        0, on_bracket
                    )  # insert whether the beetle was on the bracket at the beginning of the list
                    frame_count += 1
                return events

            proximity_incidents = []
            proximities, proximity_incidents = proximity_detection(trajectories)
            logger.info(
                "Found: [{}] proximity incidents".format(len(proximity_incidents))
            )
            incidents.extend(proximity_incidents)

            trial_incidents = TrialIncident(
                self.frame2videoTime(0), self.frame2videoTime(len(trajectories))
            )
            logger.info("Found: [1] trial incidents")
            incidents.append(trial_incidents)

            first_move_incidents = []
            first_move_incidents.append(
                find_first_move(trajectories, 0)
            )  # beetle 0 is black
            first_move_incidents.append(
                find_first_move(trajectories, 1)
            )  # beetle 1 is white
            logger.info(
                "Found: [{}] first-move incidents".format(len(first_move_incidents))
            )
            incidents.extend(first_move_incidents)

            bracket_incidents = []
            bracket_incidents.extend(get_bracket_incidents(trajectories, 0))
            bracket_incidents.extend(get_bracket_incidents(trajectories, 1))
            logger.info("Found: [{}] bracket incidents".format(len(bracket_incidents)))
            incidents.extend(bracket_incidents)
            return proximities, incidents

        def smooth_jumps(trajectories):
            pass
            # def running_mean(x, N):
            #     cumsum = np.cumsum(np.insert(x, 0, 0))
            #     return (cumsum[N:] - cumsum[:-N]) / float(N)

            # means = [[running_mean(trajectories[:,0,0], 10), running_mean(trajectories[:,0,1], 10)].T, [running_mean(trajectories[:,1,0], 10), running_mean(trajectories[:,1,1], 10)].T]
            # logger.info(np.array(trajectories) - means)
            # # N = 10;
            # # cumsum, moving_aves = [[0, 0, 0, 0]], [[]]
            # # for i, [[black_x, black_y], [white_x, white_y]] in enumerate(trajectories, 1):
            # #     cumsum.append( cumsum[i-1] + [black_x, black_y, white_x, white_y])
            # #     if i >= N:
            # #         moving_ave = (cumsum[i] - cumsum[i-N])/N

            # return trajectories

        def save_inqscribe(filename, events):
            current_directory = os.getcwd()
            logger.info(current_directory)
            events.insert(
                0, (InqscribeEvent(self.frame2videoTime(0), "Note:" + filename, None))
            )
            f = open(filename + ".inqscr", mode="w")
            f.write("app=InqScribe\n")
            event_str = ""
            for event in events:
                event_str += str(event) + "\\r"
            f.write("text={}".format(event_str))
            f.close()

        def make_video(
            trajectories, proximities, inqscribe_events: list, video_name, draw_paths
        ):
            trail = []
            vidcap = cv2.VideoCapture(self.videoFile)
            success, image = vidcap.read()
            height, width = image.shape[:2]
            out = cv2.VideoWriter(
                video_name,
                cv2.VideoWriter_fourcc("M", "J", "P", "G"),
                Job.VIDEO_OUTPUT_FPS,
                (width, height),
            )
            vidcap.set(cv2.CAP_PROP_POS_FRAMES, self.startFrame)
            for index, (point, distance) in enumerate(
                zip(tqdm(trajectories), proximities)
            ):
                success, frame = vidcap.read()
                if success:
                    # Bracket Box
                    frame = cv2.rectangle(
                        frame,
                        (
                            self.bracketROI[0] - Job.BRACKET_BUFFER,
                            self.bracketROI[1] - Job.BRACKET_BUFFER,
                        ),
                        (
                            self.bracketROI[0]
                            + self.bracketROI[2]
                            + Job.BRACKET_BUFFER,
                            self.bracketROI[1]
                            + self.bracketROI[3]
                            + Job.BRACKET_BUFFER,
                        ),
                        (100, 255, 0),
                        2,
                    )

                    # Annotate the video if the point is non-nan
                    if not np.isnan(getBlack(point)[0]) and not np.isnan(
                        getBlack(point)[1]
                    ):
                        frame = cv2.circle(
                            frame,
                            (int(getBlack(point)[0]), int(getBlack(point)[1])),
                            2,
                            (0, 0, 0),
                            3,
                        )
                    if not np.isnan(getWhite(point)[0]) and not np.isnan(
                        getWhite(point)[1]
                    ):
                        frame = cv2.circle(
                            frame,
                            (int(getWhite(point)[0]), int(getWhite(point)[1])),
                            2,
                            (255, 255, 255),
                            3,
                        )

                    font = cv2.FONT_HERSHEY_SIMPLEX
                    bottomLeftCornerOfText = (40, 40)
                    fontScale = 0.75
                    fontColor = (255, 0, 0)  # BGR
                    thickness = 1
                    distance_str = (
                        f"{distance:.0f}" if not math.isnan(distance) else "nan"
                    )
                    cv2.putText(
                        frame,
                        f"Dist: {distance_str}",
                        bottomLeftCornerOfText,
                        font,
                        fontScale,
                        fontColor,
                        thickness,
                        cv2.LINE_AA,
                    )

                    cv2.putText(
                        frame,
                        "On Bracket: {} {}".format(
                            (
                                "Black"
                                if inBox(
                                    getBlack(point), self.bracketROI, Job.BRACKET_BUFFER
                                )
                                else ""
                            ),
                            (
                                "White"
                                if inBox(
                                    getWhite(point), self.bracketROI, Job.BRACKET_BUFFER
                                )
                                else ""
                            ),
                        ),
                        (40, 70),
                        font,
                        fontScale,
                        fontColor,
                        thickness,
                        cv2.LINE_AA,
                    )
                    if distance < self.proximity_range:
                        frame = cv2.circle(
                            frame,
                            (int(getBlack(point)[0]), int(getBlack(point)[1])),
                            self.proximity_range,
                            (0, 0, 255),
                            2,
                        )
                    # add annotations to frame

                    cv2.putText(
                        frame,
                        "{}".format(str(self.frame2videoTime(index)).split(".", 2)[0]),
                        (40, 100),
                        font,
                        fontScale,
                        fontColor,
                        thickness,
                        cv2.LINE_AA,
                    )
                    if len(inqscribe_events):
                        cv2.putText(
                            frame,
                            "Past:{}".format(inqscribe_events[0]),
                            (40, 130),
                            font,
                            fontScale,
                            fontColor,
                            thickness,
                            cv2.LINE_AA,
                        )
                    if len(inqscribe_events) > 1:
                        cv2.putText(
                            frame,
                            "Next:{}".format(inqscribe_events[1]),
                            (40, 160),
                            font,
                            fontScale,
                            fontColor,
                            thickness,
                            cv2.LINE_AA,
                        )
                        if (
                            self.frame2videoTime(index + 16)
                            >= (inqscribe_events[1]).getTime()
                        ):
                            inqscribe_events.pop(0)
                    if len(inqscribe_events) > 2:
                        cv2.putText(
                            frame,
                            "     {}".format(inqscribe_events[2]),
                            (40, 190),
                            font,
                            fontScale,
                            fontColor,
                            thickness,
                            cv2.LINE_AA,
                        )

                    # Draw the beetle paths, if requested
                    def convert_location(loc):
                        """Converts a location to a tuple, using (-1, -1) for invalid locations."""
                        if np.isnan(loc[0]) or np.isnan(loc[1]):
                            return (-1, -1)
                        return (round(loc[0]), round(loc[1]))

                    def draw_line(frame, start, end, color, thickness):
                        """Draws a line if both start and end points are valid."""
                        if start != (-1, -1) and end != (-1, -1):
                            return cv2.line(
                                frame, start, end, color=color, thickness=thickness
                            )
                        return frame

                    if draw_paths:
                        black_tuple = convert_location(getBlack(point))
                        white_tuple = convert_location(getWhite(point))

                        trail.append([black_tuple, white_tuple])

                        for i in range(1, len(trail)):
                            frame = draw_line(
                                frame,
                                trail[i - 1][0],
                                trail[i][0],
                                color=(0, 0, 0),
                                thickness=2,
                            )
                            frame = draw_line(
                                frame,
                                trail[i - 1][1],
                                trail[i][1],
                                color=(255, 255, 255),
                                thickness=2,
                            )

                    out.write(frame)
                else:
                    break

            # When everything done, release the video capture and video write objects
            vidcap.release()
            out.release()

            # Closes all the frames
            cv2.destroyAllWindows()

        def save_csv(trajectories: list, proximities: dict, file_name: str) -> None:
            with open(file_name, mode="w") as csv_file:
                data_writer = csv.writer(
                    csv_file,
                    delimiter=",",
                    quotechar='"',
                    quoting=csv.QUOTE_MINIMAL,
                    lineterminator="\n",
                )
                data_writer.writerow(
                    [
                        "seconds",
                        "Black x",
                        "Black y",
                        "White x",
                        "White y",
                        "In Proximity",
                        "Black on bracket",
                        "White on bracket",
                    ]
                )
                time = 0
                for index in range(len(trajectories)):
                    time = time + Job.SECONDS_PER_FRAME
                    data_writer.writerow(
                        [
                            time,
                            getBlack(trajectories[index])[0],
                            getBlack(trajectories[index])[1],
                            getWhite(trajectories[index])[0],
                            getWhite(trajectories[index])[1],
                            proximities[index],
                            inBox(
                                getBlack(trajectories[index]),
                                self.bracketROI,
                                Job.BRACKET_BUFFER,
                            ),
                            inBox(
                                getWhite(trajectories[index]),
                                self.bracketROI,
                                Job.BRACKET_BUFFER,
                            ),
                        ]
                    )

        # Load in trajectories 'without gaps'
        self.trajectories_wo_gaps_path = (
            Path(f"session_{j.videoFile.split('.')[0]}")
            / "trajectories"
            / "without_gaps.npy"
        )
        trajectories_dict: dict = np.load(
            self.trajectories_wo_gaps_path, allow_pickle=True
        ).item()

        all_trajectories: list = trajectories_dict["trajectories"]
        logger.info(f"Trajectorys loaded len: ({len(all_trajectories)}). first index:")
        logger.info(all_trajectories[0])

        # Crop the trajectories array down to the analyzed part
        trajectories: list = all_trajectories[self.startFrame : self.videoEndFrame - 1]
        logger.info("---------- INTERPOLATION ----------")
        trajectories = verify_and_swap_black_white(interpolate(trajectories))
        # trajectories = smooth_jumps(trajectories)

        logger.info("---------- INCIDENT ANALYSIS ----------")
        proximities, incidents = find_all_incidents(trajectories)
        logger.info("TOTAL: [{}] incidents found".format(len(incidents)))

        logger.info("---------- DATA EXPORT ----------")
        save_csv(trajectories, proximities, "post_proc_output.csv")
        logger.info("Position data saved to: {}".format("post_proc_output.csv"))

        inqscribe_events = []
        for incident in incidents:
            for event in incident.inqscribe_events:
                logger.info(f"Event: {event}")
                inqscribe_events.append(event)
        inqscribe_events.sort()
        save_inqscribe(self.path, inqscribe_events)
        logger.info("Inqscribe file saved to: {}".format("post_proc_output.csv"))

        make_video(
            trajectories,
            proximities,
            inqscribe_events,
            "post_proc_output.avi",
            self.options["visualize_paths"],
        )
        logger.info("Annotated video saved to {}".format("post_proc_output.avi"))


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="[%(levelname)s] %(filename)s:%(lineno)d: %(message)s",
    )
    j = Job(sys.argv[1])

    start_cwd = os.getcwd()
    os.chdir(j.getDirectory())

    if j.options["do_tracking"]:
        cmd_result = os.system(j.cmd)
        logger.info(f"Tracking result: {cmd_result}")
        logger.info("Tracking complete.")
    else:
        logger.info("Skipping tracking.")

    if j.options["do_postprocessing"]:
        j.postprocess()
        logger.info("Postprocessing complete.")
    else:
        logger.info("Skipping postprocessing.")

    os.chdir(start_cwd)
