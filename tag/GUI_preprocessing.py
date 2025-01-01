import tkinter as tk
from tkinter import filedialog, LabelFrame
import cv2
from PIL import ImageTk, Image
from preprocessing import Video
import os


class Runner:

    class Statuses:
        def __init__(self):
            self.video = None
            self.bracket = None
            self.arena = None
            self.beetle = None
            self.start = None
            self.end = None

    def updateStartEnds(self, *args):
        self.statuses.start = self.startFrameVar.get()
        self.statuses.end = self.endFrameVar.get()
        try:
            self.currentVideo.startFrame = int(self.statuses.start)
            self.currentVideo.videoEndFrame = int(self.statuses.end)
        except:
            print("Failure to interpret frame range")

    def __init__(self):
        root = tk.Tk()
        root.title("Beetle Tracker v0.2.0")
        self.currentVideo = Video()
        self.statuses = self.Statuses()
        self.savePath = os.getcwd()
        title = tk.Label(text="Beetle Tracker GUI", padx=80, pady=20)

        self.newVideoBtn = tk.Button(
            text="Select Video",
            command=lambda: self.currentVideo.initialize(self.newVideo()),
        )
        videoTxt = tk.Label(text="Video: ", pady=8)
        self.videoStatus = tk.Label(text="None", fg="Red")

        self.selectBracketBtn = tk.Button(
            root,
            text="Select Bracket",
            state="disabled",
            command=self.selectBracketBtnFn,
        )
        bracketTxt = tk.Label(text="Bracket: ", pady=8)
        self.bracketStatus = tk.Label(text="None", fg="Red")

        self.selectArenaBtn = tk.Button(
            text="Select Arena", state="disabled", command=self.selectArenaBtnFn
        )
        arenaTxt = tk.Label(text="Arena: ", pady=8)
        self.arenaStatus = tk.Label(text="None", fg="Red")

        self.selectBeetleBtn = tk.Button(
            text="Select Beetle",
            state="disabled",
            command=self.getBeetleSelectWrapperFn,
        )
        beetleTxt = tk.Label(text="Black beetle: ", pady=8)
        self.beetleStatus = tk.Label(text="None", fg="Red")

        self.startFrameVar = tk.StringVar()
        self.endFrameVar = tk.StringVar()
        self.startFrameVar.trace("w", self.updateStartEnds)
        self.endFrameVar.trace("w", self.updateStartEnds)

        self.selectStartBtn = tk.Button(
            text="Set Start", state="disabled", command=self.getFirstFrameWrapperFn
        )
        startTxt = tk.Label(text="Start Frame: ", pady=8)
        self.startStatus = tk.Entry(
            text="None", fg="Red", width=4, textvariable=self.startFrameVar
        )

        self.selectEndBtn = tk.Button(
            text="Set End", state="disabled", command=self.getLastFrameWrapperFn
        )
        endTxt = tk.Label(text="End Frame: ", pady=8)
        self.endStatus = tk.Entry(
            text="None", fg="Red", width=4, textvariable=self.endFrameVar
        )

        instructions = tk.Label(text="Select a video file to start")

        self.startButton = tk.Button(
            text="Save Files",
            state="disabled",
            padx=10,
            pady=10,
            command=lambda: self.startFn(self.savePath),
        )

        self.savePathBtn = tk.Button(text="Output", command=self.changeSavePathFn)
        self.savePathTxt = tk.Label(text="{}".format(self.savePath), pady=8)

        title.grid(columnspan=3)

        self.newVideoBtn.grid(column=0, row=1)
        videoTxt.grid(sticky=tk.W, column=1, row=1)
        self.videoStatus.grid(sticky=tk.W, column=2, row=1)

        self.selectBracketBtn.grid(column=0, row=2)
        bracketTxt.grid(sticky=tk.W, column=1, row=2)
        self.bracketStatus.grid(sticky=tk.W, column=2, row=2)

        self.selectArenaBtn.grid(column=0, row=3)
        arenaTxt.grid(sticky=tk.W, column=1, row=3)
        self.arenaStatus.grid(sticky=tk.W, column=2, row=3)

        self.selectStartBtn.grid(column=0, row=4)
        startTxt.grid(sticky=tk.W, column=1, row=4)
        self.startStatus.grid(sticky=tk.W, column=2, row=4)

        self.selectEndBtn.grid(column=0, row=5)
        endTxt.grid(sticky=tk.W, column=1, row=5)
        self.endStatus.grid(sticky=tk.W, column=2, row=5)

        self.selectBeetleBtn.grid(column=0, row=6)
        beetleTxt.grid(sticky=tk.W, column=1, row=6)
        self.beetleStatus.grid(sticky=tk.W, column=2, row=6)

        self.startButton.grid(columnspan=3, row=7)
        instructions.grid(columnspan=3, row=8)

        self.savePathBtn.grid(column=0, row=9)
        self.savePathTxt.grid(column=1, columnspan=2, row=9)
        root.mainloop()

    def selectBracketBtnFn(self):
        res = self.currentVideo.getBracketROIWrapper()
        if res:
            self.statuses.bracket = "Done"
        else:
            self.statuses.bracket = None
        self.updateStatuses()

    def selectArenaBtnFn(self):
        res = self.currentVideo.getArenaROIWrapper()
        if res:
            self.statuses.arena = "Done"
        else:
            self.statuses.arena = None
        self.updateStatuses()

    def getFirstFrameWrapperFn(self):
        res = self.currentVideo.getFirstFrameWrapper()
        if res:
            self.statuses.start = str(res)
            self.startFrameVar.set(str(res))
        else:
            self.statuses.start = None
        self.updateStatuses()

    def getLastFrameWrapperFn(self):
        res = self.currentVideo.getLastFrameWrapper()
        if res:
            self.statuses.end = str(res)
            self.endFrameVar.set(str(res))
        else:
            self.statuses.end = None
        self.updateStatuses()

    def getBeetleSelectWrapperFn(self):
        res = self.currentVideo.getBeetleSelectWrapper()
        if res:
            self.statuses.beetle = str(res)
        else:
            self.statuses.beetle = None
        self.updateStatuses()

    def changeSavePathFn(self):
        dirname = tk.filedialog.askdirectory(
            initialdir="../", title="Select a directory"
        )
        self.savePath = dirname
        self.updateStatuses()

    def newVideo(self):
        def pickFile():
            path = tk.filedialog.askopenfilename(
                initialdir="../",
                title="Select file",
                filetypes=(("movie files", "*.mp4"), ("all files", "*.*")),
            )
            testCapture = cv2.VideoCapture(path)
            if not testCapture.isOpened():
                return pickFile()
            else:
                ret, frame = testCapture.read()
                if not ret:
                    return pickFile()

            print("Loaded in video {} successfully.".format(path))
            return path

        f = pickFile()
        print(f)
        self.statuses.video = os.path.basename(f)
        self.updateStatuses()
        return f

    def updateStatuses(self):
        if self.statuses.video:
            self.videoStatus["text"] = self.statuses.video
            self.videoStatus["fg"] = "Green"

            self.selectBracketBtn["state"] = "normal"
            self.selectArenaBtn["state"] = "normal"
        else:
            self.videoStatus["fg"] = "red"
            self.videoStatus["text"] = "None"
            self.selectBracketBtn["state"] = "disabled"
            self.selectArenaBtn["state"] = "disabled"
            self.selectStartBtn["state"] = "disabled"
            self.selectEndBtn["state"] = "disabled"

        if self.statuses.bracket:
            self.bracketStatus["fg"] = "green"
            self.bracketStatus["text"] = "Done"
        else:
            self.bracketStatus["fg"] = "red"
            self.bracketStatus["text"] = "None"

        if self.statuses.arena:
            self.arenaStatus["fg"] = "green"
            self.arenaStatus["text"] = "Done"
            self.selectStartBtn["state"] = "normal"
            self.selectEndBtn["state"] = "normal"
        else:
            self.arenaStatus["fg"] = "red"
            self.arenaStatus["text"] = "None"
            self.selectStartBtn["state"] = "disabled"
            self.selectEndBtn["state"] = "disabled"

        if self.statuses.start:
            self.startStatus["fg"] = "green"
            self.startFrameVar.set(str(self.statuses.start))
            self.selectBeetleBtn["state"] = "normal"
        else:
            self.startStatus["fg"] = "red"
            self.startFrameVar.set("0")
            self.selectBeetleBtn["state"] = "disabled"

        if self.statuses.end:
            self.endStatus["fg"] = "green"
            self.endFrameVar.set(str(self.statuses.end))
        else:
            self.endStatus["fg"] = "red"
            self.endFrameVar.set("0")

        if self.statuses.beetle:
            self.beetleStatus["fg"] = "green"
            self.beetleStatus["text"] = self.statuses.beetle
        else:
            self.beetleStatus["fg"] = "red"
            self.beetleStatus["text"] = "None"

        if (
            self.statuses.video
            and self.statuses.bracket
            and self.statuses.arena
            and self.statuses.start
            and self.statuses.end
            and self.statuses.beetle
        ):
            self.startButton["state"] = "normal"
        else:
            self.startButton["state"] = "disabled"

        self.savePathTxt["text"] = self.savePath + "/" + self.statuses.video

    def startFn(self, savePath):
        path = self.currentVideo.savePostProcData(savePath)
        print("Directory generated at {}".format(path))
        os.startfile(path)


if __name__ == "__main__":
    r = Runner()
