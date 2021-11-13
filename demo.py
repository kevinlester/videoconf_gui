import argparse
import os.path
import time
from collections import deque
from dataclasses import dataclass
from threading import Thread, Lock
from tkinter import *
from tkinter import ttk

import cv2
import numpy as np
import torch
from PIL import Image, ImageTk
from torch import nn
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

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
    def __init__(self, camera, labelframe):
        self.camera = camera
        self.labelframe = labelframe
        self.fps_tracker = FPSTracker(ratio=0.05)

    def __del__(self):
        print(f"{self.name()} destructing.")

    def name(self): pass
    def activate(self): pass
    def read_frame(self): pass


class NoopFrameManipulator(FrameManipulator):
    def name(self):
        return "noop"

    def read_frame(self):
        frame = self.camera.read()
        self.fps_tracker.tick()
        return frame, self.fps_tracker.get()


class FgSeparatorManipulator(FrameManipulator):
    class BGProcessor:
        def __init__(self, args, label_frame):
            self.args = args
            self.label_frame = label_frame
            self.orig_bg = None

        def captured_bg(self, frame):
            self.orig_bg = frame

        def process_bg(self, pha, fgr): pass

        def name(self): pass

    class BlurBGProcessor(BGProcessor):
        def __init__(self, args, label_frame):
            super().__init__(args, label_frame)
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
        def __init__(self, args, label_frame):
            super().__init__(args, label_frame)
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
        def __init__(self, args, label_frame):
            super().__init__(args, label_frame)
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

        def __init__(self, args, label_frame):
            super().__init__(args, label_frame)
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

    class FGProcessor:
        def __init__(self, args, label_frame):
            self.args = args
            self.label_frame = label_frame
            chkValue = BooleanVar()
            chkValue.set(False)
            self.chk_button = Checkbutton(self.label_frame,
                                          text=self.name(),
                                          var=chkValue)
            self.chk_button.pack()
            self.chkValue = chkValue

        def name(self): pass
        def is_enabled(self): return self.chkValue.get()
        def process_fg(self, pha, fgr): pass

    class GhostFGProcessor(FGProcessor):
        def __init__(self, args, label_frame):
            super().__init__(args, label_frame)
            self.scale_control = Scale(label_frame, from_=0, to=10, orient=HORIZONTAL, tickinterval=1, length=250)
            self.scale_control.set(10)
            self.scale_control.pack()

        def name(self):
            return "Ghost"

        def process_fg(self, pha, fgr):
            pha[0,0,:,:] *= self.scale_control.get() / 10.0
            return pha, fgr

    class HologramFGProcessor(FGProcessor):
        def __init__(self, args, label_frame):
            super().__init__(args, label_frame)
            self.band_length_scale = Scale(label_frame, from_=1, to=20, orient=HORIZONTAL, tickinterval=1, length=250)
            self.band_length_scale.set(2)
            self.band_length_scale.pack()

            self.band_gap_scale = Scale(label_frame, from_=1, to=20, orient=HORIZONTAL, tickinterval=1, length=250)
            self.band_gap_scale.set(3)
            self.band_gap_scale.pack()


        def process_fg(self, pha, fgr):
            return pha, self.frame_to_hologram(pha, fgr)

        def name(self):
            return "Hologram"

        def frame_to_hologram(self, pha, fgr):
            mask = pha * fgr
            mask_img = mask.mul(255).byte().cpu().permute(0, 2, 3, 1).numpy()[0]
            mask_img = cv2.cvtColor(mask_img, cv2.COLOR_RGB2BGR)
            mask_img = self.hologram_effect(mask_img)
            mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2RGB)
            return cv2_frame_to_cuda(mask_img)

        def hologram_effect(self, img):
            # add a blue tint
            holo = cv2.applyColorMap(img, cv2.COLORMAP_WINTER)
            # add a halftone effect
            # band_length, band_gap = 2, 3
            band_length, band_gap = self.band_length_scale.get(), self.band_gap_scale.get()

            for y in range(holo.shape[0]):
                if y % (band_length + band_gap) < band_length:
                    holo[y, :, :] = holo[y, :, :] * np.random.uniform(0.1, 0.3)

            # add some ghosting
            holo_blur = cv2.addWeighted(holo, 0.2, self.shift_image(holo.copy(), 5, 5), 0.8, 0)
            holo_blur = cv2.addWeighted(holo_blur, 0.4, self.shift_image(holo.copy(), -5, -5), 0.6, 0)

            # combine with the original color, oversaturated
            out = cv2.addWeighted(img, 0.5, holo_blur, 0.6, 0)
            return out

        def shift_image(self, img, dx, dy):
            img = np.roll(img, dy, axis=0)
            img = np.roll(img, dx, axis=1)
            if dy > 0:
                img[:dy, :] = 0
            elif dy < 0:
                img[dy:, :] = 0
            if dx > 0:
                img[:, :dx] = 0
            elif dx < 0:
                img[:, dx:] = 0
            return img

    def grab_background(self):
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

    def __init__(self, camera, labelframe, args):
        super().__init__(camera, labelframe)
        self.thread = None
        self.read_lock = Lock()
        self.thread = None
        self.current_frame = None
        self.current_fps = 0
        self.is_active = False

        self.state = 'background'
        self.bgr_frame = None
        self.bgmModel = BGModel(args.model_checkpoint, args.model_backbone_scale, args.model_refine_mode,
                                args.model_refine_sample_pixels, args.model_refine_threshold)
        self.bg_processor = None
        self.bg_processors = [
            self.BlurBGProcessor(args, labelframe),
            self.WhiteBGProcessor(args, labelframe),
            self.StaticBGProcessor(args, labelframe),
            self.OrigBGProcessor(args, labelframe)
        ]
        static_processor = self.StaticBGProcessor(args, labelframe)
        if static_processor.is_valid:
            self.bg_processors.append(static_processor)

        video_processor = self.VideoBGProcessor(args, labelframe)
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

        self.fg_processors = [
            self.GhostFGProcessor(args, self.labelframe),
            self.HologramFGProcessor(args, self.labelframe)
        ]

    def name(self):
        return "Foreground Separator"

    def activate(self):
        print("activating")
        self.bgmModel.reload()
        self.thread = Thread(target=self.process_frames, args=())
        self.thread.daemon = True
        self.thread.start()

    def process_frames(self):
        while True:
            frame = self.camera.read()
            frame = self.process_frame(frame)
            with self.read_lock:
                self.current_fps = self.fps_tracker.tick()
                self.current_frame = frame

    def process_frame(self, frame):
        if self.bgr_frame is None:
            return frame

        cuda_frame = cv2_frame_to_cuda(frame)
        pha, fgr = self.bgmModel.model(cuda_frame, self.bgr_frame)[:2]

        tgt_bgr = self.bg_processor.process_bg(pha, fgr)

        for fg_processor in self.fg_processors:
            if fg_processor.is_enabled():
                pha, fgr = fg_processor.process_fg(pha, fgr)

        res = pha * fgr + (1 - pha) * tgt_bgr
        res = res.mul(255).byte().cpu().permute(0, 2, 3, 1).numpy()[0]
        return res

    def read_frame(self):
        # BUG!  Need to store the fps and framerate locally for a correct framarate
        with self.read_lock:
            frame, fps = self.current_frame.copy(), self.current_fps
        return frame, fps


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
        self.deque = deque(maxlen=2)
        self.last_frame = None

    def __update(self):
        while self.success_reading:
            grabbed, frame = self.capture.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # with self.read_lock:
            #     self.success_reading = grabbed
            #     self.frame = frame
            self.deque.append(frame)

    def read(self):
        # with self.read_lock:
        #     frame = self.frame.copy()
        # return frame
        try:
            frame = self.deque.popleft()
            frame = frame.copy()
            self.last_frame = frame
        except IndexError:
            # print("Empty queue")
            frame = self.last_frame
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
    # frame = camera.read()

    cv2image, manipulator_rate = frame_manipulator.read_frame()
    if cv2image is None:
        print("No frame")
        img_label.after(1, show_frames)
        return

    # send the image through the manipulator
    # cv2image = frame_manipulator.process_frame(frame)

    # Convert frame to PhotoImage for display
    # can try imgtk.paste to see if that is faster
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    img_label.imgtk = imgtk
    img_label.configure(image=imgtk)

    # if self.webcam is not None:
    #     image_web = np.ascontiguousarray(image, dtype=np.uint8) # .copy()
    #     image_web = cv2.cvtColor(image_web, cv2.COLOR_RGB2BGR)
    #     self.webcam.schedule_frame(image_web)

    fps_tracker.tick()
    display_fps = fps_tracker.get() or 0
    manipulator_rate = manipulator_rate or 0
    fps_label.configure(text=f'{display_fps:.2f} {manipulator_rate:.2f}')

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


def set_manipulator():
    global frame_manipulator
    frame_manipulator = FgSeparatorManipulator(camera, option_frame, args)
    frame_manipulator.activate()
    time.sleep(1)


if __name__ == '__main__':
    args = load_args()

    root = Tk()
    width, height = args.resolution
    source_device_id = 0

    camera = Camera(device_id=args.source_device_id, width=width, height=height)
    fps_tracker = FPSTracker(ratio=0.05)

    p = ttk.Panedwindow(orient=HORIZONTAL)
    p.pack(fill=BOTH, expand=1)
    # two panes, each of which would get widgets gridded into it:
    f1 = ttk.Labelframe(p, text='camrea', width=width, height=height)
    img_label = Label(f1, text='hi')
    img_label.grid(row=0, column=0)

    f2 = ttk.Labelframe(p, text='Pane2', width=250, height=100)
    fps_label = Label(f2, text='0')
    fps_label.pack()
    manipulator_btn = Button(f2, text="Foreground Manipulator", command=set_manipulator)
    manipulator_btn.pack()
    option_frame = ttk.Labelframe(f2, text='Options', width=250, height=100)
    option_frame.pack()

    frame_manipulator = NoopFrameManipulator(camera, option_frame)
    frame_manipulator.activate()

    p.add(f1)
    p.add(f2)
    time.sleep(1)
    show_frames()
    root.mainloop()
