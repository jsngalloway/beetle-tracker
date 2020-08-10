import tkinter as tk
from tkinter import filedialog, LabelFrame
import cv2
from PIL import ImageTk, Image
from preprocessing import Video
import os

class Runner:
    def __init__(self):
        root = tk.Tk()
        self.currentVideo = Video()
        title = tk.Label(text="Beetle Tracker GUI", padx=80, pady=20)
        # btnGroup = LabelFrame(root, padx=5, pady=5)

        self.newVideoBtn = tk.Button(text="Select Video", command=lambda: self.currentVideo.initialize(self.newVideo()))
        videoTxt = tk.Label(text="Video: ", pady=8)
        self.videoStatus = tk.Label(text="None", fg="Red")

        self.selectBracketBtn = tk.Button(root, text="Select Bracket", state="disabled", command=self.selectBracketBtnFn)
        bracketTxt = tk.Label(text="Bracket: ", pady=8)
        self.bracketStatus = tk.Label(text="None", fg="Red")

        self.selectArenaBtn = tk.Button(text="Select Arena", state="disabled", command=self.selectArenaBtnFn)
        arenaTxt = tk.Label(text="Arena: ", pady=8)
        self.arenaStatus = tk.Label(text="None", fg="Red")

        self.selectBeetleBtn = tk.Button(text="Select Beetle", state="disabled", command=self.getBeetleSelectWrapperFn)
        beetleTxt = tk.Label(text="Black beetle: ", pady=8)
        self.beetleStatus = tk.Label(text="None", fg="Red")

        self.selectStartBtn = tk.Button(text="Set Start", state="disabled", command=self.getFirstFrameWrapperFn)
        startTxt = tk.Label(text="Start Frame: ", pady=8)
        self.startStatus = tk.Label(text="None", fg="Red")

        self.selectEndBtn = tk.Button(text="Set End", state="disabled", command=self.getLastFrameWrapperFn)
        endTxt = tk.Label(text="End Frame: ", pady=8)
        self.endStatus = tk.Label(text="None", fg="Red")
        
        instructions = tk.Label(text="Select a video file to start")

        startButton = tk.Button(text="Save Files", state="disabled", padx=10, pady=10)

        title.grid(columnspan=3)

        self.newVideoBtn.grid(column=0, row=1)
        videoTxt.grid(sticky = tk.W, column=1, row=1)
        self.videoStatus.grid(sticky = tk.W, column=2, row=1)

        self.selectBracketBtn.grid(column=0, row=2)
        bracketTxt.grid(sticky = tk.W, column=1, row=2)
        self.bracketStatus.grid(sticky = tk.W, column=2, row=2)

        self.selectArenaBtn.grid(column=0, row=3)
        arenaTxt.grid(sticky = tk.W, column=1, row=3)
        self.arenaStatus.grid(sticky = tk.W, column=2, row=3)

        self.selectBeetleBtn.grid(column=0, row=4)
        beetleTxt.grid(sticky = tk.W, column=1, row=4)
        self.beetleStatus.grid(sticky = tk.W, column=2, row=4)

        self.selectStartBtn.grid(column=0, row=5)
        startTxt.grid(sticky = tk.W, column=1, row=5)
        self.startStatus.grid(sticky = tk.W, column=2, row=5)

        self.selectEndBtn.grid(column=0, row=6)
        endTxt.grid(sticky = tk.W, column=1, row=6)
        self.endStatus.grid(sticky = tk.W, column=2, row=6)
        
        startButton.grid(columnspan=3)
        instructions.grid(columnspan=3)
        root.mainloop()

    def activateAllButtons(self):
        self.selectBracketBtn["state"] = "normal"
        self.selectArenaBtn["state"] = "normal"
        self.selectBeetleBtn["state"] = "normal"

    def selectBracketBtnFn(self):
        res = self.currentVideo.getBracketROIWrapper()
        if self.currentVideo.bracketROI:
            self.bracketStatus["fg"] = "green"
            self.bracketStatus["text"] = "Done"

    def selectArenaBtnFn(self):
        res = self.currentVideo.getArenaROIWrapper()
        if res:
            self.arenaStatus["fg"] = "green"
            self.arenaStatus["text"] = "Done"
            self.selectStartBtn["state"] = "normal"
            self.selectEndBtn["state"] = "normal"
    
    def getFirstFrameWrapperFn(self):
        res = self.currentVideo.getFirstFrameWrapper()
        if res:
            self.startStatus["fg"] = "green"
            self.startStatus["text"] = str(res)

    def getLastFrameWrapperFn(self):
        res = self.currentVideo.getLastFrameWrapper()
        if res:
            self.endStatus["fg"] = "green"
            self.endStatus["text"] = str(res)
    def getBeetleSelectWrapperFn(self):
        res = self.currentVideo.getBeetleSelectWrapper()
        if res:
            self.beetleStatus["fg"] = "green"
            self.beetleStatus["text"] = str(res)


    def newVideo(self):
        def pickFile():
            path =  tk.filedialog.askopenfilename(initialdir = "../",title = "Select file",filetypes = (("movie files","*.mp4"),("all files","*.*")))
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
        self.activateAllButtons()
        self.videoStatus["text"] = os.path.basename(f)
        self.videoStatus["fg"] = "Green"

        return f
        # self.currentVideo = Video(f)

    
    # def getBeetleBracketBtnFn(self):


if __name__ == '__main__' :
    r = Runner()
