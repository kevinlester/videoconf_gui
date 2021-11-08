import os.path
from tkinter import *
from tkinter import ttk
from dataclasses import dataclass
import torch
from torch import nn
from torchvision.transforms import Compose, ToTensor, Resize
from torch.utils.data import Dataset, DataLoader

import numpy as np
import argparse
import cv2
import time
from threading import Thread, Lock
from PIL import Image, ImageTk

camera = -1
img_label = -1
frame_manipulator = None


@dataclass
class BGModel:
    model_checkpoint: str
    backbone_scale: float
    refine_mode: str
    refine_sample_pixels: int
    refine_threshold: float

    def model(self):
        return self.model

    def reload(self):
        self.model = torch.jit.load(self.model_checkpoint)
        self.model.backbone_scale = self.backbone_scale
        self.model.refine_mode = self.refine_mode
        self.model.refine_sample_pixels = self.refine_sample_pixels
        self.model.model_refine_threshold = self.refine_threshold
        # print(self.refine_mode + " , " + str(self.refine_sample_pixels) + ", " +
        #       str(self.refine_threshold)+ ", " + str(self.backbone_scale))
        self.model.cuda().eval()


class FrameManipulator:
    def __init__(self, labelframe):
        self.labelframe = labelframe

    def name(self): pass
    def activate(self): pass
    def process_frame(self, frame): pass


class NoopFrameManipulator(FrameManipulator):
    def name(self):
        return "noop"

    def process_frame(self, frame):
        return frame


class FgSeparatorManipulator(FrameManipulator):
    class BGProcessor:
        def __init__(self, args):
            self.args = args
            self.orig_bg = None

        def captured_bg(self, frame):
            self.orig_bg = frame

        def process_bg(self, pha, fgr): pass

        def name(self): pass

    class BlurBGProcessor(BGProcessor):
        def __init__(self, args):
            super().__init__(args)
            self.bg = None

        def name(self):
            return "Blur"

        def captured_bg(self, frame):
            print("capture")
            bgr_blur = cv2.GaussianBlur(frame.astype('float32'), (67, 67), 0).astype('uint8')
            # cv2.blur(bgr_frame, (10, 10))
            self.bg = cv2_frame_to_cuda(bgr_blur)

        def process_bg(self, pha, fgr):
            return self.bg

    class WhiteBGProcessor(BGProcessor):
        def __init__(self, args):
            super().__init__(args)
            self.bg = None

        def name(self):
            return "White"

        def process_bg(self, pha, fgr):
            return torch.ones_like(fgr)

    class OrigBGProcessor(BGProcessor):
        def name(self):
            return "Original"

        def captured_bg(self, frame):
            self.orig_bg = cv2_frame_to_cuda(frame)

        def process_bg(self, pha, fgr):
            return self.orig_bg

    class StaticBGProcessor(BGProcessor):
        def __init__(self, args):
            super().__init__(args)
            if args.target_image is None or not os.path.exists(args.target_image):
                self.is_valid = False
                return
            self.is_valid = True
            frame = cv2.cvtColor(cv2.imread(args.target_image), cv2.COLOR_BGR2RGB)
            self.bg = cv2_frame_to_cuda(frame)

        def name(self):
            return "Static Image"

        def process_bg(self, pha, fgr):
            return nn.functional.interpolate(self.bg, (fgr.shape[2:]))

    class VideoBGProcessor(BGProcessor):

        class VideoDataset(Dataset):
            def __init__(self, path: str, transforms: any = None):
                self.cap = cv2.VideoCapture(path)
                self.transforms = transforms

                self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                self.frame_rate = self.cap.get(cv2.CAP_PROP_FPS)
                self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

            def __len__(self):
                return self.frame_count

            def __getitem__(self, idx):
                if isinstance(idx, slice):
                    return [self[i] for i in range(*idx.indices(len(self)))]

                if self.cap.get(cv2.CAP_PROP_POS_FRAMES) != idx:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, img = self.cap.read()
                if not ret:
                    raise IndexError(f'Idx: {idx} out of length: {len(self)}')
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
                if self.transforms:
                    img = self.transforms(img)
                return img

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_value, exc_traceback):
                self.cap.release()

        def __init__(self, args):
            super().__init__(args)
            if args.target_image is None or not os.path.exists(args.target_image):
                self.is_valid = False
                return
            self.is_valid = True
            self.target_background_frame = 0
            self.tb_video = self.VideoDataset(args.target_video, transforms=ToTensor())

        def name(self):
            return "Video"

        def process_bg(self, pha, fgr):
            vidframe = self.tb_video[self.target_background_frame].unsqueeze_(0).cuda()
            tgt_bgr = nn.functional.interpolate(vidframe, (fgr.shape[2:]))
            self.target_background_frame += 1
            if self.target_background_frame >= self.tb_video.__len__():
                self.target_background_frame = 0
            return tgt_bgr

    def grab_background(self):
        print("grabBackground")
        if self.bg_processor is not None:
            self.bg_processor = None
            return
        frame = camera.read()
        self.bgr_frame = cv2_frame_to_cuda(frame)
        for bg_processor in self.bg_processors:
            bg_processor.captured_bg(frame)
        self.bg_processor = self.bg_processors[self.bg_combobox.current()]
        self.state = self.bg_processor.name()

    def bg_processor_changed(self, event_object):
        self.bg_processor = self.bg_processors[self.bg_combobox.current()]

    def __init__(self, labelframe, args):
        super().__init__(labelframe)
        self.state = 'background'
        self.bgr_frame = None
        self.bgmModel = BGModel(args.model_checkpoint, args.model_backbone_scale, args.model_refine_mode,
                                args.model_refine_sample_pixels, args.model_refine_threshold)
        self.bg_processor = None
        self.bg_processors = [
            self.BlurBGProcessor(args),
            self.WhiteBGProcessor(args),
            self.StaticBGProcessor(args),
            self.OrigBGProcessor(args)
        ]
        static_processor = self.StaticBGProcessor(args)
        if static_processor.is_valid:
            self.bg_processors.append(static_processor)

        video_processor = self.VideoBGProcessor(args)
        if video_processor.is_valid:
            self.bg_processors.append(video_processor)

        bg_options = []
        for bg_processor in self.bg_processors:
            bg_options.append(bg_processor.name())

        bg_combobox = ttk.Combobox(labelframe, values=bg_options, height=200)
        bg_combobox.current(0)
        bg_combobox.pack()
        bg_combobox.bind("<<ComboboxSelected>>", self.bg_processor_changed)
        self.bg_combobox = bg_combobox

        self.background_btn = Button(self.labelframe, text="Grab Background", command=self.grab_background, width="250")
        self.background_btn.pack()

    def name(self):
        return "Foreground Separator"

    def activate(self):
        print("activating")
        self.bgmModel.reload()

    def process_frame(self, frame):
        if self.bg_processor is None:
            return frame

        cuda_frame = cv2_frame_to_cuda(frame)
        pha, fgr = self.bgmModel.model(cuda_frame, self.bgr_frame)[:2]

        tgt_bgr = self.bg_processor.process_bg(pha, fgr)

        res = pha * fgr + (1 - pha) * tgt_bgr
        res = res.mul(255).byte().cpu().permute(0, 2, 3, 1).numpy()[0]
        return res


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
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            with self.read_lock:
                self.success_reading = grabbed
                self.frame = frame

    def read(self):
        with self.read_lock:
            frame = self.frame.copy()
        return frame

    def __exit__(self, exec_type, exc_value, traceback):
        self.capture.release()


class FPSTracker:
    def __init__(self, ratio=0.5):
        self._last_tick = None
        self._avg_fps = None
        self.ratio = ratio

    def tick(self):
        if self._last_tick is None:
            self._last_tick = time.time()
            return None
        t_new = time.time()
        fps_sample = 1.0 / (t_new - self._last_tick)
        self._avg_fps = self.ratio * fps_sample + (1 - self.ratio) * self._avg_fps \
            if self._avg_fps is not None else fps_sample
        self._last_tick = t_new
        return self.get()

    def get(self):
        return self._avg_fps


def show_frames():
    # Get the latest frame and convert into Image
    frame = camera.read()

    cv2image = frame_manipulator.process_frame(frame)
    img = Image.fromarray(cv2image)
    # Convert image to PhotoImage
    imgtk = ImageTk.PhotoImage(image=img)
    img_label.imgtk = imgtk
    img_label.configure(image=imgtk)

    # if self.webcam is not None:
    #     image_web = np.ascontiguousarray(image, dtype=np.uint8) # .copy()
    #     image_web = cv2.cvtColor(image_web, cv2.COLOR_RGB2BGR)
    #     self.webcam.schedule_frame(image_web)

    fps_tracker.tick()
    fps_label.configure(text=fps_tracker.get())
    # Repeat after an interval to capture continuously
    img_label.after(1, show_frames)


def cv2_frame_to_cuda(frame):
    return ToTensor()(Image.fromarray(frame)).unsqueeze_(0).cuda()


def load_args():
    parser = argparse.ArgumentParser(description='Video Conference demo')

    parser.add_argument('--model-backbone-scale', type=float, default=0.25)
    parser.add_argument('--model-checkpoint', type=str, required=False)
    parser.add_argument('--model-checkpoint-dir', type=str, required=False)

    parser.add_argument('--model-refine-mode', type=str, default='sampling',
                        choices=['full', 'sampling', 'thresholding'])
    parser.add_argument('--model-refine-sample-pixels', type=int, default=80_000)
    parser.add_argument('--model-refine-threshold', type=float, default=0.7)

    parser.add_argument('--hide-fps', action='store_true')
    parser.add_argument('--resolution', type=int, nargs=2, metavar=('width', 'height'), default=(1280, 720))
    parser.add_argument('--target-video', type=str, default='./demo_video.mp4')
    parser.add_argument('--target-image', type=str, default='./demo_image.jpg')
    parser.add_argument('--camera-device', type=str, default='/dev/video1')
    parser.add_argument('--source_device_id', type=int, default=0)
    return parser.parse_args()


def set_manipupator():
    global frame_manipulator
    frame_manipulator = FgSeparatorManipulator(option_frame, args)
    frame_manipulator.activate()


if __name__ == '__main__':
    args = load_args()

    root = Tk()
    width, height = args.resolution
    source_device_id = 0

    camera = Camera(device_id=args.source_device_id, width=width, height=height)
    fps_tracker = FPSTracker()

    p = ttk.Panedwindow(orient=HORIZONTAL)
    p.pack(fill=BOTH, expand=1)
    # two panes, each of which would get widgets gridded into it:
    f1 = ttk.Labelframe(p, text='camrea', width=width, height=height)
    img_label = Label(f1, text='hi')
    img_label.grid(row=0, column=0)

    f2 = ttk.Labelframe(p, text='Pane2', width=250, height=100)
    fps_label = Label(f2, text='0')
    fps_label.pack()
    manipulator_btn = Button(f2, text="Foreground Manipulator", command=set_manipupator)
    manipulator_btn.pack()
    option_frame = ttk.Labelframe(f2, text='Options', width=250, height=100)
    option_frame.pack()

    frame_manipulator = NoopFrameManipulator(option_frame)
    frame_manipulator.activate()

    p.add(f1)
    p.add(f2)

    show_frames()
    root.mainloop()
