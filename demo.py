from tkinter import *
from tkinter import ttk
import numpy as np
import argparse
import cv2
from threading import Thread, Lock
from PIL import Image, ImageTk

camera = -1
img_label = -1


# A wrapper that reads data from cv2.VideoCapture in its own thread to optimize.
# Use .read() in a tight loop to get the newest frame
class Camera:
    def __init__(self, device_id=0, width=1280, height=720):
        self.capture = cv2.VideoCapture(device_id)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        self.success_reading, self.frame = self.capture.read()
        print(self.success_reading)
        self.read_lock = Lock()
        self.thread = Thread(target=self.__update, args=())
        self.thread.daemon = True
        self.thread.start()

    def __update(self):
        while self.success_reading:
            grabbed, frame = self.capture.read()
            with self.read_lock:
                self.success_reading = grabbed
                self.frame = frame

    def read(self):
        with self.read_lock:
            frame = self.frame.copy()
        return frame

    def __exit__(self, exec_type, exc_value, traceback):
        self.capture.release()


def show_frames():
    # Get the latest frame and convert into Image
    cv2image = cv2.cvtColor(camera.read(), cv2.COLOR_BGR2RGB)
    img = Image.fromarray(cv2image)
    # Convert image to PhotoImage
    imgtk = ImageTk.PhotoImage(image=img)
    img_label.imgtk = imgtk
    img_label.configure(image=imgtk)
    # Repeat after an interval to capture continiously
    img_label.after(1, show_frames)


def load_args():
    parser = argparse.ArgumentParser(description='Video Conference demo')

    parser.add_argument('--resolution', type=int, nargs=2, metavar=('width', 'height'), default=(1280, 720))
    parser.add_argument('--source_device_id', type=int, default=0)
    return parser.parse_args()


if __name__ == '__main__':
    args = load_args()

    root = Tk()
    width, height = args.resolution
    source_device_id = 0

    camera = Camera(device_id=args.source_device_id, width=width, height=height)

    p = ttk.Panedwindow(orient=HORIZONTAL)
    p.pack(fill=BOTH, expand=1)
    # two panes, each of which would get widgets gridded into it:
    f1 = ttk.Labelframe(p, text='camrea', width=width, height=height)
    f2 = ttk.Labelframe(p, text='Pane2', width=100, height=100)
    img_label = Label(f1, text='hi')
    img_label.grid(row=0, column=0)
    p.add(f1)
    p.add(f2)

    show_frames()
    root.mainloop()
