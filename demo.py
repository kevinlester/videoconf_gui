import argparse
import os.path
import time
from collections import deque
from dataclasses import dataclass
import multiprocessing as mp
from multiprocessing import Process
from threading import Thread, Lock
from tkinter import *
from tkinter import ttk
from queue import Empty

import cv2
import numpy as np
import torch
from PIL import Image, ImageTk
from torch import nn
from torch.jit import ScriptModule
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

camera = -1
img_label = -1
frame_manipulator = None


class FrameManipulator:
    def __init__(self, camera, labelframe):
        self.camera = camera
        self.labelframe = labelframe
        self.fps_tracker = FPSTracker(ratio=0.05)

    def __del__(self):
        print(f"{self.name()} destructing.")
        if self.camera is not None:
            self.camera.close()
        self.camera = None

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

    class BaseFGUIManager:
        def __init__(self, args, parent, label_frame, parent_listener_conn):
            self.args = args
            self.parent = parent
            self.label_frame = label_frame
            self.parent_listener_conn = parent_listener_conn

        def configure_ui(self):
            chk_value = BooleanVar()
            chk_value.set(False)
            chk_button = Checkbutton(self.label_frame,
                                     text=self.parent.name(),
                                     var=chk_value,
                                     command=lambda name="is_enabled": self.gui_updated(name, chk_value.get()))
            chk_button.pack()

        def gui_updated(self, name, value):
            print("%s changed to %s" % (name, value))
            self.parent_listener_conn.send({"attribute": [self.parent.name(), name, value]})

    class GhostUIManager(BaseFGUIManager):
        def __init__(self, args, parent, label_frame, parent_listener_conn):
            super().__init__(args, parent, label_frame, parent_listener_conn)
            self.scale_control = None

        def configure_ui(self):
            super().configure_ui()
            scale_control = Scale(self.label_frame,
                                  from_=0,
                                  to=10,
                                  orient=HORIZONTAL,
                                  tickinterval=1,
                                  length=250,
                                  command=lambda value, name="fade_value": self.gui_updated(name, float(value)))
            scale_control.set(10)
            scale_control.pack()
            self.scale_control = scale_control

    class HologramUIManager(BaseFGUIManager):
        def __init__(self, args, parent, label_frame, parent_listener_conn):
            super().__init__(args, parent, label_frame, parent_listener_conn)
            self.band_length_scale = None
            self.band_gap_scale = None

        def configure_ui(self):
            super().configure_ui()
            band_length_scale = Scale(self.label_frame, from_=1, to=20, orient=HORIZONTAL, tickinterval=1, length=250,
                                      command=lambda value, name="band_length": self.gui_updated(name, int(value)))
            band_length_scale.set(2)
            band_length_scale.pack()

            band_gap_scale = Scale(self.label_frame, from_=1, to=20, orient=HORIZONTAL, tickinterval=1, length=250,
                                   command=lambda value, name="band_gap": self.gui_updated(name, int(value)))
            band_gap_scale.set(3)
            band_gap_scale.pack()
            self.band_length_scale = band_length_scale
            self.band_gap_scale = band_gap_scale

    class ModelProcessor(Process):

        @dataclass
        class BGModel:
            model_checkpoint: str
            backbone_scale: float
            refine_mode: str
            refine_sample_pixels: int
            refine_threshold: float
            model: ScriptModule

            def reload(self):
                self.model = torch.jit.load(self.model_checkpoint)
                self.model.backbone_scale = self.backbone_scale
                self.model.refine_mode = self.refine_mode
                self.model.refine_sample_pixels = self.refine_sample_pixels
                self.model.model_refine_threshold = self.refine_threshold
                # print(self.refine_mode + " , " + str(self.refine_sample_pixels) + ", " +
                #       str(self.refine_threshold)+ ", " + str(self.backbone_scale))
                self.model.cuda().eval()

        class BGProcessor:
            def __init__(self, args):
                self.args = args
                self.orig_bg = None

            def captured_bg(self, frame):
                self.orig_bg = frame

            def process_bg(self, pha, fgr): pass

            @staticmethod
            def name(): pass

            @staticmethod
            def configure_gui():
                pass

        class BlurBGProcessor(BGProcessor):
            def __init__(self, args):
                super().__init__(args)
                self.bg = None

            @staticmethod
            def name():
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

            @staticmethod
            def name():
                return "White"

            def process_bg(self, pha, fgr):
                return torch.ones_like(fgr)

        class OrigBGProcessor(BGProcessor):
            @staticmethod
            def name():
                return "Original"

            def captured_bg(self, frame):
                self.orig_bg = cv2_frame_to_cuda(frame)

            def process_bg(self, pha, fgr):
                return self.orig_bg

        class StaticBGProcessor(BGProcessor):
            def __init__(self, args):
                super().__init__(args)
                if not self.is_valid(args):
                    self.is_valid = False
                    return
                self.is_valid = True
                frame = cv2.cvtColor(cv2.imread(args.target_image), cv2.COLOR_BGR2RGB)
                self.bg = cv2_frame_to_cuda(frame)

            @staticmethod
            def name():
                return "Static Image"

            #TODO: is this still needed?
            @staticmethod
            def is_valid(args):
                return args.target_image is not None and os.path.exists(args.target_image)

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
                if not self.is_valid(args):
                    self.is_valid = False
                    return
                self.is_valid = True
                self.target_background_frame = 0
                self.tb_video = self.VideoDataset(args.target_video, transforms=ToTensor())

            @staticmethod
            def name():
                return "Video"

            @staticmethod
            def is_valid(args):
                return args.target_video is not None and os.path.exists(args.target_video)

            def process_bg(self, pha, fgr):
                vidframe = self.tb_video[self.target_background_frame].unsqueeze_(0).cuda()
                tgt_bgr = nn.functional.interpolate(vidframe, (fgr.shape[2:]))
                self.target_background_frame += 1
                if self.target_background_frame >= self.tb_video.__len__():
                    self.target_background_frame = 0
                return tgt_bgr

        class FGProcessor:

            def __init__(self, args):
                self.args = args
                self.is_enabled = False

            @staticmethod
            def name(): pass

            def process_fg(self, pha, fgr): pass

            @staticmethod
            def create_ui_manager(args, label_frame, parent_listener_conn): pass

        class GhostFGProcessor(FGProcessor):

            def __init__(self, args):
                super().__init__(args)
                self.fade_value = 10

            @staticmethod
            def name(): return "Ghost"

            def process_fg(self, pha, fgr):
                pha[0, 0, :, :] *= self.fade_value / 10.0
                return pha, fgr

            @staticmethod
            def create_ui_manager(args, label_frame, parent_listener_conn):
                return FgSeparatorManipulator.GhostUIManager(args,
                                                             FgSeparatorManipulator.ModelProcessor.GhostFGProcessor,
                                                             label_frame,
                                                             parent_listener_conn)

        class HologramFGProcessor(FGProcessor):
            def __init__(self, args):
                super().__init__(args)
                self.band_length = 2
                self.band_gap = 3

            def process_fg(self, pha, fgr):
                return pha, self.frame_to_hologram(pha, fgr)

            @staticmethod
            def name(): return "Hologram"

            @staticmethod
            def create_ui_manager(args, label_frame, parent_listener_conn):
                return FgSeparatorManipulator.HologramUIManager(args,
                                                                FgSeparatorManipulator.ModelProcessor.HologramFGProcessor,
                                                                label_frame,
                                                                parent_listener_conn)

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
                band_length, band_gap = self.band_length, self.band_gap

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

        def __start_listener(self):
            self.listener_thread = Thread(target=self.__listen, args=())
            self.listener_thread.daemon = True
            self.listener_thread.start()

        def __listen(self):
            while True:
                listener_conn = self.listener_conn
                listener_conn.poll(None)
                msg_dict = listener_conn.recv()
                print("Got message")
                with self.lock:
                    self.msg_dict = msg_dict
                    self.msg_present = True

        def cv2_frame_to_cuda(self, frame):
            return ToTensor()(Image.fromarray(frame)).unsqueeze_(0).cuda()

        def process_msg(self):
            msg_dict = self.msg_dict
            if "command" in msg_dict:
                cmd = msg_dict["command"][0]
                if cmd == "GRAB_BG":
                    frame = self.camera.read()
                    self.bgr_frame = self.cv2_frame_to_cuda(frame)
                    for bg_processor in self.bg_processors.values():
                        bg_processor.captured_bg(frame)
                if cmd == "CHANGE_BG":
                    processor_name = msg_dict["command"][1]
                    self.bg_processor = self.bg_processors[processor_name]
            if "attribute" in msg_dict:
                processor, attribute, value = msg_dict["attribute"]
                setattr(self.fg_processors[processor], attribute, value)

            self.msg_present = False

        def process_frame(self, frame):
            if self.bgr_frame is None:
                return frame

            cuda_frame = self.cv2_frame_to_cuda(frame)
            pha, fgr = self.bgmModel.model(cuda_frame, self.bgr_frame)[:2]

            tgt_bgr = self.bg_processor.process_bg(pha, fgr)

            for fg_processor in self.fg_processors.values():
                if fg_processor.is_enabled:
                    pha, fgr = fg_processor.process_fg(pha, fgr)

            res = pha * fgr + (1 - pha) * tgt_bgr
            res = res.mul(255).byte().cpu().permute(0, 2, 3, 1).numpy()[0]
            return res

        def __init__(self, args, conn, listener_conn):
            super(FgSeparatorManipulator.ModelProcessor, self).__init__()
            print("initializing")
            self.daemon = True
            self.args = args
            self.fps_tracker = None
            self.conn = conn
            self.listener_conn = listener_conn
            self.camera = None
            self.listener_thread = None
            self.lock = None
            self.msg_present = False
            self.bgr_frame = None
            self.bg_processor = None
            self.bgmModel = None
            self.bg_processors = None
            self.fg_processors = None

        def run(self):
            self.lock = Lock()
            self.fps_tracker = FPSTracker(ratio=0.05)
            self.camera = Camera(device_id=0, width=1280, height=720)
            self.bgmModel = self.BGModel(self.args.model_checkpoint,
                                         self.args.model_backbone_scale,
                                         self.args.model_refine_mode,
                                         self.args.model_refine_sample_pixels,
                                         self.args.model_refine_threshold,
                                         None)
            self.bgmModel.reload()

            self.bg_processors = {
                self.BlurBGProcessor.name(): self.BlurBGProcessor(self.args),
                self.WhiteBGProcessor.name(): self.WhiteBGProcessor(self.args),
                self.StaticBGProcessor.name(): self.StaticBGProcessor(self.args),
                self.OrigBGProcessor.name(): self.OrigBGProcessor(self.args)
            }

            static_processor = self.StaticBGProcessor(self.args)
            if static_processor.is_valid:
                self.bg_processors[self.StaticBGProcessor.name()] = static_processor

            video_processor = self.VideoBGProcessor(self.args)
            if video_processor.is_valid:
                self.bg_processors[self.VideoBGProcessor.name()] = video_processor

            self.bg_processor = self.bg_processors[self.BlurBGProcessor.name()]

            self.fg_processors = {
                self.GhostFGProcessor.name(): self.GhostFGProcessor(self.args),
                self.HologramFGProcessor.name() : self.HologramFGProcessor(self.args)
            }
            self.__start_listener()

            conn = self.conn
            # TODO - remove this by reading the first frame from the camera.
            time.sleep(1)
            while True:
                with self.lock:
                    if self.msg_present:
                        self.process_msg()

                frame = self.camera.read()
                frame = self.process_frame(frame)

                fps = self.fps_tracker.tick()
                #print(f"manipulator fps = {self.current_fps}")

                try:
                    #print("adding to queue")
                    #conn.send((frame, fps))
                    #bytes = frame.tobytes()
                    #print(f"sent {len(bytes)} bytes. shape = {frame.shape} dtype = {frame.dtype}")

                    #arr = bytearray(frame.tobytes())
                    #arr.append(fps)
                    #conn.send_bytes(arr)

                    conn.send_bytes(frame.tobytes())

                    #conn.send(fps)
                    #print(fps)
                    #print("addied to queue")
                except:
                    print("queue unexpectedly full")

    def grab_background(self):
        self.parent_listener_conn.send({"command": ["GRAB_BG"]})

        bg_options = []
        for bg_processor in self.bg_processors:
            bg_options.append(bg_processor.name())

        if self.bg_combobox is not None:
            return

        bg_combobox = ttk.Combobox(self.label_frame, values=bg_options, height=200)
        bg_combobox.current(0)
        bg_combobox.pack()
        bg_combobox.bind("<<ComboboxSelected>>", self.bg_processor_changed)
        self.bg_combobox = bg_combobox

        for fg_ui_manager in self.fg_ui_managers:
            fg_ui_manager.configure_ui()

    def bg_processor_changed(self, event_object):
        self.parent_listener_conn.send({"command": ["CHANGE_BG", event_object.widget.get()]})

    def __init__(self, camera, label_frame, args):
        super().__init__(camera, label_frame)
        self.current_frame = None
        self.current_fps = 0
        self.last_frame = None
        self.last_fps = 0
        self.is_active = False
        self.label_frame = label_frame
        self.parent_conn = None
        self.parent_listener_conn = None
        self.bg_process = None
        self.bg_combobox = None

        self.bg_processors = [
            self.ModelProcessor.BlurBGProcessor,
            self.ModelProcessor.WhiteBGProcessor,
            self.ModelProcessor.OrigBGProcessor
        ]

        static_processor = self.ModelProcessor.StaticBGProcessor
        if static_processor.is_valid(args):
            self.bg_processors.append(static_processor)

        video_processor = self.ModelProcessor.VideoBGProcessor
        if video_processor.is_valid(args):
            self.bg_processors.append(video_processor)

        self.background_btn = Button(self.labelframe, text="Grab Background", command=self.grab_background, width="250")
        self.background_btn.pack()

        self.fg_ui_managers = None

    def name(self):
        return "Foreground Separator"

    def activate(self):
        print("activating")
        parent_conn, child_conn = mp.Pipe()
        parent_listener_conn, child_listener_conn = mp.Pipe()
        self.parent_conn = parent_conn
        self.parent_listener_conn = parent_listener_conn

        self.fg_ui_managers = [
            self.ModelProcessor.GhostFGProcessor.create_ui_manager(args, self.label_frame, parent_listener_conn),
            self.ModelProcessor.HologramFGProcessor.create_ui_manager(args, self.label_frame, parent_listener_conn)
        ]

        self.bg_process = self.ModelProcessor(args, child_conn, child_listener_conn)
        self.bg_process.start()

    bytebuffer = bytearray(1280*720*3*np.dtype(np.uint8).itemsize)

    def read_frame(self):
        counter = 0
        frame = self.last_frame
        fps = self.last_fps

        while self.parent_conn.poll() and counter < 10:
            try:
                #frame, fps = self.parent_conn.recv()

                recv_bytes = self.parent_conn.recv_bytes()
                frame = np.frombuffer(recv_bytes, dtype=np.uint8).reshape((720, 1280, 3))
                #fps = 0 #self.parent_conn.recv()

                #self.parent_conn.recv_bytes_into(FgSeparatorManipulator.bytebuffer)
                #frame = np.frombuffer(FgSeparatorManipulator.bytebuffer, dtype=np.uint8).reshape((720, 1280, 3))

                #print(f"received {len(recv_bytes)} bytes. shape = {frame.shape}")

                counter += 1
                #print("got queue")
            except Empty:
                #print("empty queue")
                frame = self.last_frame
                fps = self.last_fps
                break
        self.last_frame = frame
        self.last_fps = fps
        #if counter > 1: print(f"got {counter} frames")
        return frame, fps

        # while not self.queue.empty() and counter < 10:
        #     try:
        #         frame = self.queue.get_nowait()
        #         counter += 1
        #         # print("got queue")
        #     except Empty:
        #         print("empty queue")
        #         frame = self.last_frame
        #         break
        # self.last_frame = frame
        # if counter > 1: print(f"got {counter} frames")
        # return frame, 0
        # BUG!  Need to store the fps and framerate locally for a correct framarate
        # with self.read_lock:
        #     frame, fps = self.current_frame.copy(), self.current_fps
        # return frame, fps


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
        self.read_lock = Lock()
        self.thread = Thread(target=self.__update, args=())
        self.thread.daemon = True
        self.thread.start()
        self.deque = deque(maxlen=2)
        self.last_frame = None

    def __update(self):
        while True:
            grabbed, frame = self.capture.read()
            with self.read_lock:
                if not (self.success_reading and grabbed):
                    break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.deque.append(frame)
        print("done looping")

    def read(self):
        if self.deque:
            frame = self.deque.popleft()
            #frame = frame.copy()
            self.last_frame = frame
        else:
            frame = self.last_frame
        return frame

    def __exit__(self, exec_type, exc_value, traceback):
        self.capture.release()
        print("camera exiting")

    def close(self):
        with self.read_lock:
            self.success_reading = False
        self.thread.join()
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
        #print("No frame")
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

def close_window():
    print("User requested close at:", time.time())
    print("Destroying GUI at:", time.time())
    try: # "destroy()" can throw, so you should wrap it like this.
        root.destroy()
    except:
        pass


if __name__ == '__main__':
    mp.set_start_method('spawn')
    args = load_args()

    root = Tk()
    root.protocol("WM_DELETE_WINDOW", close_window)
    width, height = args.resolution
    source_device_id = 0

    camera = Camera(width=width, height=height, device_id=args.source_device_id)
    fps_tracker = FPSTracker(ratio=0.05)

    p = ttk.Panedwindow(orient=HORIZONTAL, )
    p.pack(fill=BOTH, expand=1)
    # two panes, each of which would get widgets gridded into it:
    f1 = ttk.Labelframe(p, text='camera', width=width, height=height)
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
    # TODO - remove this by reading the first frame from the camera.
    time.sleep(1)
    show_frames()
    root.mainloop()
