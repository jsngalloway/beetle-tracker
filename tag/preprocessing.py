import os
import sys
from typing import Tuple
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


def addInstructionsToImage(img, *args):
    img_copy = np.copy(img)
    font = cv2.FONT_HERSHEY_PLAIN
    bottomLeftCornerOfText = [30, 30]
    fontScale = 1.125
    fontColor = (255, 0, 0)  # BGR
    lineType = 1
    for text in args:
        cv2.putText(
            img_copy,
            text,
            tuple(bottomLeftCornerOfText),
            font,
            fontScale,
            fontColor,
            lineType,
        )
        bottomLeftCornerOfText[1] = bottomLeftCornerOfText[1] + 20
    return img_copy


class Video:
    SESSION_NAME = "script"
    START_TIME = 20 * 1000  # 20 seconds, the time we initially jump to
    BRACKET_BUFFER = 10
    WINDOW_NAME = "window"

    def __init__(self):
        pass

    def initialize(self, p):
        self.path = p
        # TODO this is arbitrary
        self.proximity_range = 45
        self.vidcap = cv2.VideoCapture(self.path)
        self.videoFile = os.path.basename(self.path)

        self.length = int(self.vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.time = Video.START_TIME
        self.vidcap.set(cv2.CAP_PROP_POS_MSEC, self.time)
        success, self.image = self.vidcap.read()
        self.height, self.width = self.image.shape[:2]

        self.bracketROI = None
        self.basicArenaROI = None
        self.arenaROIstr = None
        self.black = None
        self.startFrame = None
        self.videoEndFrame = None

        if not success:
            print(
                "Video failed to load. Verify file is a good video file or contact Jesse."
            )
            print(self.path)
            exit(1)

    def getBracketROIWrapper(self):
        self.bracketROI = self.getBracketROI(self.image, "Bracket Selection")
        return len(self.bracketROI)

    def getArenaROIWrapper(self):
        self.basicArenaROI, self.arenaROIstr = self.getArenaROI2(
            self.image, "Arena Selection"
        )
        return len(self.basicArenaROI)

    def getBeetleSelectWrapper(self):
        self.black = self.beetleSelect(self.image, "Beetle Selection")
        return self.black

    def getFirstFrameWrapper(self):
        self.startFrame = self.getFirstFrame("First Frame Selection")
        return self.startFrame

    def getLastFrameWrapper(self):
        self.videoEndFrame = self.getLastFrame("Last Frame Selection")
        return self.videoEndFrame

    def finishAndSave(self):
        cv2.destroyAllWindows()

        # Print out data to show successful save
        print(
            "Processing Complete.\n\tStart Frame: {}/{}\n\tArena: {}\n\tBracket (x,y,w,h): {}".format(
                self.startFrame, self.length, self.arenaROIstr, self.bracketROI
            )
        )
        self.savePostProcData()

    def getTrackingCommand(self):
        return f'idtrackerai --track --video_paths "{self.videoFile}" --intensity_ths 0 162 --area_ths 150 60000 --tracking_intervals [{self.startFrame},{self.videoEndFrame}] --number_of_animals 2 --roi_list "{self.arenaROIstr}"'

    def beetleSelect(self, img, windowName):
        img = addInstructionsToImage(img, "Click near black beetle. Enter to continue")
        print("Click to select black beetle. Press any key to continue...", end="")

        class CoordinateStore:
            def __init__(self, image):
                self.black = []
                self.old_image = np.copy(image)
                self.image = image

            def select_point(self, event, x, y, flags, param):
                if event == cv2.EVENT_LBUTTONDOWN:
                    if not self.black == []:
                        self.black = []
                        cv2.imshow(windowName, self.old_image)
                        self.image = np.copy(self.old_image)
                    self.black = [x, y]
                    self.image = cv2.circle(self.image, (x, y), 13, (0, 0, 0), 2)
                    cv2.imshow(windowName, self.image)

            def getBlack(self):
                return self.black

        coordStore = CoordinateStore(img)

        cv2.imshow(windowName, img)
        cv2.setMouseCallback(windowName, coordStore.select_point)
        k = None
        # accept space or enter to continue
        while not ((k == 32) or (k == 13)):
            k = cv2.waitKey(50)
        cv2.setMouseCallback(windowName, lambda *args: None)
        retVal = coordStore.getBlack()
        cv2.destroyWindow(windowName)
        print("done.")
        return retVal

    def getArenaROI2(self, img, windowName) -> Tuple[tuple, str]:
        """
        Given an image, walks the user through selecting and masking the Beetle arena
        Parameters:
            img - the image to display used for masking
            windowName - the cv2 name of the window
        Returns:
            tuple ROI - the basic rectangular ROI of the initial Arena boxing
            str - the masked contour ROI converted to a string format
        """

        def contourToString(contour):
            """
            Accepts a contour and converts it into a string formatted for idtrackerai
            """
            point_lst = ["[["]
            first = True
            for point in contour:
                if not first:
                    point_str = ",({},{})".format(point[0][0], point[0][1])
                else:
                    point_str = "({},{})".format(point[0][0], point[0][1])
                    first = False
                point_lst.append(point_str)
            point_lst.append("]]")
            return "".join(point_lst)

        class MaskManager:
            """
            Class MaskManager Creates a mask of the same size of the video
            It's hooked in as a callback and processes mouse clicks to update the mask
            """

            def __init__(self, img):
                self.mask = np.zeros(img.shape[:2], np.uint8)
                self.updated = False

            def draw_circle(self, event, x, y, flags, param):
                if event == cv2.EVENT_LBUTTONUP:
                    # Left click marks as foreground
                    for i in range(y - 10, y + 10):
                        for j in range(x - 10, x + 10):
                            self.mask[i][j] = cv2.GC_FGD
                    self.updated = True

                if event == cv2.EVENT_RBUTTONUP:
                    # Right click marks as background
                    for i in range(y - 10, y + 10):
                        for j in range(x - 10, x + 10):
                            self.mask[i][j] = cv2.GC_BGD
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
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)

        print("Selecting Arena ROI...", end="")
        rect = (0, 0, 0, 0)
        while rect == (0, 0, 0, 0):
            rect = cv2.selectROI(
                windowName,
                addInstructionsToImage(
                    img,
                    "Drag to select entire Arena.",
                    "Any key to continue, drag again to reset",
                ),
            )

        cv2.imshow(windowName, addInstructionsToImage(img, "Finding arena..."))
        cv2.waitKey(1)

        # Initialize the contours using the rectangle we drew using selectROI
        msk, bgdModel, fgdModel = cv2.grabCut(
            img, mm.getMask(), rect, bgdModel, fgdModel, 3, cv2.GC_INIT_WITH_RECT
        )
        mm.setMask(msk)
        contours = None
        cv2.setMouseCallback(windowName, mm.draw_circle)

        while 1:
            # If mask==2 or mask== 1, mask2 get 0, other wise it gets 1 as 'uint8' type.
            mask2 = np.where((mm.getMask() == 2) | (mm.getMask() == 0), 0, 1).astype(
                "uint8"
            )
            img_cut = img * mask2[:, :, np.newaxis]

            imgray = cv2.cvtColor(img_cut, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(imgray, 127, 255, 0)

            contours = None
            find_contours_result = cv2.findContours(
                thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )
            if len(find_contours_result) == 2:
                contours = find_contours_result[0]
            elif len(find_contours_result) == 3:
                contours = find_contours_result[1]
            else:
                print("Unexpected findContour size!")

            largest = 0
            for i in range(len(contours)):
                if cv2.contourArea(contours[i]) > cv2.contourArea(contours[largest]):
                    largest = i

            disp_img = np.copy(img_cut)
            cv2.drawContours(disp_img, contours, largest, (0, 0, 255), 2)

            while 1:
                invert_mask = 1 - mm.getMask()
                disp_img[:, :, 1] = cv2.bitwise_and(
                    img_cut, img_cut, mask=mm.getMask() * 255
                )[:, :, 1]
                disp_img[:, :, 2] = cv2.bitwise_and(
                    img_cut, img_cut, mask=invert_mask * 255
                )[:, :, 2]
                cv2.drawContours(disp_img, contours, largest, (255, 0, 0), 2)
                cv2.imshow(
                    windowName,
                    addInstructionsToImage(
                        disp_img,
                        "Left-click = mark Arena",
                        "Right-click = mark background",
                        "Enter to continue",
                    ),
                )

                k = cv2.waitKey(1)
                if not k == -1:
                    break

            # If the user hits enter without making an update we break
            if not mm.getUpdateFlag():
                break
            else:
                # The user made an update so we'll re-generate the model
                cv2.imshow(
                    windowName,
                    addInstructionsToImage(disp_img, "Re-generating contours..."),
                )
                cv2.waitKey(1)
                mm.resetUpdateFlag()
                msk, bgdModel, fgdModel = cv2.grabCut(
                    img,
                    mm.getMask(),
                    None,
                    bgdModel,
                    fgdModel,
                    3,
                    cv2.GC_INIT_WITH_MASK,
                )
                mm.setMask(msk)

        # Remove the callback we assigned earlier
        cv2.setMouseCallback(windowName, lambda *args: None)

        cv2.destroyWindow(windowName)
        print("done.")
        return (rect, contourToString(contours[largest]))

    def getBracketROI(self, img, windowName) -> tuple:
        """
        Prompts the user to select the fungus bracket
        Parameters:
            img - the image to display
            windowName - the name of the window to display it in
        Returns:
            tuple bracket - the ROI of the bracket
        """
        print("Select the bracket....", end="")
        bracket = (0, 0, 0, 0)
        while bracket == (0, 0, 0, 0):
            bracket = cv2.selectROI(
                windowName,
                addInstructionsToImage(
                    self.image,
                    "Select the fungus bracket.",
                    "Enter to continue, drag again to reset",
                ),
            )
        cv2.destroyWindow(windowName)
        print("done.")
        return bracket

    def getFirstFrame(self, windowName):
        return self.autoGetFrame("Select First Frame", 10, 1.5, "Start")

    def getLastFrame(self, windowName):
        return self.autoGetFrame(
            "Select Last Frame",
            self.length - 400,
            0.82,
            "Near end (fr {})".format(self.length - 400),
        )

    def autoGetFrame(self, windowName, startPoint, sensitivity, trackbarName):
        strt = 0
        velocity = 5
        self.vidcap.set(cv2.CAP_PROP_POS_FRAMES, startPoint)  # move to frame 0

        def getBrightness(self, r, img):
            def getRandX(roi):
                return random.randint(r[0], r[0] + r[2])

            def getRandY(roi):
                return random.randint(r[1], r[1] + r[3])

            sum = 0
            for i in range(0, 500):
                x = getRandX(r)
                y = getRandY(r)
                sum = (
                    sum + img[y, x, 0]
                )  # I'm still not sure why x and y are swapped here
            return sum / 500

        success, img = self.vidcap.read()
        cv2.imshow(windowName, img)
        cv2.waitKey(1)

        if success:
            strt = getBrightness(self, self.basicArenaROI, img)
        else:
            print("Error autodetecting brightness.")

        autoDetectedFrame: int = None

        def onChange(trackbarValue):
            self.vidcap.set(cv2.CAP_PROP_POS_FRAMES, startPoint + trackbarValue)
            success, img = self.vidcap.read()
            cv2.imshow(
                windowName,
                addInstructionsToImage(
                    img,
                    "Selecting frame [{}/{}]".format(
                        startPoint + trackbarValue, self.length
                    ),
                    "Arrow keys prev/next frame",
                    "Enter to confirm",
                    (
                        (
                            "Detected near: "
                            + str(autoDetectedFrame - velocity + startPoint)
                            + "-"
                            + str(autoDetectedFrame + startPoint)
                        )
                        if (autoDetectedFrame)
                        else ""
                    ),
                ),
            )
            cv2.waitKey(1)

        trackbarStart = 0
        TRACKBAR_LENGTH = 400
        cv2.createTrackbar(
            trackbarName, windowName, trackbarStart, TRACKBAR_LENGTH, onChange
        )
        print("Selecting frame...", end="")
        for i in range(startPoint, startPoint + TRACKBAR_LENGTH, velocity):
            if i >= self.length - 2:
                print("Failed auto detection, proceed manually.")
                break
            self.vidcap.set(cv2.CAP_PROP_POS_FRAMES, i)
            success, image = self.vidcap.read()

            cv2.setTrackbarPos(
                trackbarName, windowName, trackbarStart + (i - startPoint)
            )

            cfrm = getBrightness(self, self.basicArenaROI, image)
            if ((sensitivity > 1) and (cfrm > strt * sensitivity)) or (
                (sensitivity <= 1) and (cfrm <= strt * sensitivity)
            ):
                autoDetectedFrame = cv2.getTrackbarPos(trackbarName, windowName)

                # Manually call onChange once to update the frame displayed
                onChange(cv2.getTrackbarPos(trackbarName, windowName))
                print("Adjust or press any key to confirm Selection...")

                k = None
                while not ((k == 32) or (k == 13) or (k == 36) or (k == 76)):
                    k = cv2.waitKeyEx()
                    if k == 2424832 or k == ord(
                        "a"
                    ):  # left arrow on windows TODO: check on Mac
                        cv2.setTrackbarPos(
                            trackbarName,
                            windowName,
                            cv2.getTrackbarPos(trackbarName, windowName) - 1,
                        )
                    if k == 2555904 or k == ord("d"):  # right arrow on windows
                        cv2.setTrackbarPos(
                            trackbarName,
                            windowName,
                            cv2.getTrackbarPos(trackbarName, windowName) + 1,
                        )

                # User has hit space or enter to confirm frame
                startFrame = startPoint + cv2.getTrackbarPos(trackbarName, windowName)
                print("Frame confirmed. {}".format(startFrame))
                cv2.destroyWindow(windowName)
                return startFrame

    def savePostProcData(self, target_area):

        options = {
            "visualize_paths": False,
            "do_tracking": True,
            "do_postprocessing": True,
        }

        video_id = os.path.splitext(os.path.basename(self.path))[0]
        target_dir = os.path.join(target_area, video_id)

        try:
            os.mkdir(target_dir)
        except OSError:
            print(
                "Creation of the directory %s failed, maybe it already exists..."
                % target_dir
            )
        else:
            print("Successfully created %s folder" % target_dir)

        f = open(os.path.join(target_dir, (video_id + ".json")), "w+")

        class JsonOut:
            def __init__(
                self,
                cmd,
                location_black,
                startFrame,
                videoEndFrame,
                proximity_range,
                bracketROI,
                videoFile,
                options,
            ):
                self.cmd = cmd
                self.location_black = location_black
                self.startFrame = startFrame
                self.videoEndFrame = videoEndFrame
                self.proximity_range = proximity_range
                self.bracketROI = bracketROI
                self.video = videoFile
                self.options = options

        jOut = JsonOut(
            cmd=self.getTrackingCommand(),
            location_black=self.black,
            startFrame=self.startFrame,
            videoEndFrame=self.videoEndFrame,
            proximity_range=self.proximity_range,
            bracketROI=self.bracketROI,
            videoFile=self.videoFile,
            options=options,
        )
        jsonStr = json.dumps(jOut, indent=4, default=lambda o: o.__dict__)
        f.write(jsonStr)
        f.close()

        # copy the original video file into the new directory
        copyfile(self.path, os.path.join(target_dir, os.path.basename(self.path)))
        print(
            "Successfully generated tracking files for {} at {}".format(
                video_id, target_dir
            )
        )

        # Ding ding
        print("\007")
        return target_dir


if __name__ == "__main__":
    print(
        "Run the processing script on GPU cluster:\npython postprocessing.py {}".format(
            v.path
        )
    )
