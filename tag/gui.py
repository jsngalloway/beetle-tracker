import tkinter as tk
import cv2
from PIL import ImageTk, Image
top = tk.Tk()

# Code to add widgets will go here...

# image = cv2.imread(path)

# Open up a video from the sample video folder with OpenCV

class VideoPlayer:
    def __init__(self, canvas, vid, original_image):
        self.canvas = canvas
        self.vid = vid
        self.original_image = original_image

    def jumpToFrame(self):
        print("JUMPPPPPPPPPPPPPPPPPPPPPPPPPPPPPppp")
        self.vid.set(cv2.CAP_PROP_POS_FRAMES, 2000)


    def nextFrame(self):
        # I have no idea why we need original image
        print("Frame: {}".format(self.original_image))
        ret, frame = self.vid.read()
        image = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        self.canvas.itemconfig(1, image = image)
        self.original_image = image
        self.canvas.after(50, self.nextFrame)


canvas = tk.Canvas(top, width = 300, height = 300)


vid = cv2.VideoCapture("../sample_videos/video1.mp4")

ret, frame = vid.read()
image = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
first_image = canvas.create_image(20, 20, image=image)
print("successfully created an image")
vp = VideoPlayer(canvas, vid, first_image)
print("first image + {}".format(first_image))
vp.nextFrame()



greeting = tk.Label(text="Beetle Tracker GUI")
button = tk.Button(text="Click me!", command=vp.jumpToFrame)
  
greeting.pack()
button.pack()
canvas.pack()

top.mainloop()