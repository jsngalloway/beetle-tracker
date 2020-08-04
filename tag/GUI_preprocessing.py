import tkinter as tk
from tkinter import filedialog, LabelFrame
import cv2
from PIL import ImageTk, Image
from preprocessing import Video

class Runner:
    def __init__(self):
        root = tk.Tk()
        self.currentVideo = Video()
        greeting = tk.Label(text="Beetle Tracker GUI")
        self.newVideoBtn = tk.Button(text="New Video", command=lambda: self.currentVideo.initialize(self.newVideo()))
        self.selectBracketBtn = tk.Button(text="Select Bracket", state="disabled", command=self.selectBracketBtnFn)
        self.selectArenaBtn = tk.Button(text="Select Arena", state="disabled", command=self.currentVideo.getArenaROIWrapper)
        self.selectBeetleBtn = tk.Button(text="Select Beetle", state="disabled", command=self.currentVideo.getBeetleSelectWrapper)
        self.selectStartEndBtn = tk.Button(text="Select Arena", state="disabled", command=self.currentVideo.getFirstFrameWrapper)

        greeting.pack()
        group = LabelFrame(root, text="Group", padx=5, pady=5)
        group.pack(padx=10, pady=10)
        self.newVideoBtn.pack()
        self.selectBracketBtn.pack()
        self.selectArenaBtn.pack()
        self.selectBeetleBtn.pack()
        self.selectStartEndBtn.pack()
        root.mainloop()

    def selectBracketBtnFn(self):
        self.currentVideo.getBracketROIWrapper()
        if self.currentVideo.bracketROI:
            self.selectArenaBtn["state"] = "normal"
        else:
            self.selectArenaBtn["state"] = "disabled"

    def newVideo(self):
        def pickFile():
            path =  tk.filedialog.askopenfilename(initialdir = "./",title = "Select file",filetypes = (("movie files","*.mp4"),("all files","*.*")))
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
        self.selectBracketBtn["state"] = "normal"
        return f
        # self.currentVideo = Video(f)

    
    # def getBeetleBracketBtnFn(self):


if __name__ == '__main__' :
    r = Runner()
