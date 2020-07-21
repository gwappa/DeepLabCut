"""
Author: Hao Wu
hwu01@g.harvard.edu

This is the helper class for video reading and saving in DeepLabCut.
Updated by AM

You can set various codecs below,
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
i.e. 'XVID'
"""

import numpy as np
import cv2
from . import frame_pickers, frame_writers

class VideoProcessor(object):
    '''
    Base class for a video processing unit, implementation is required for video loading and saving

    sh and sw are the output height and width respectively.
    '''
    def __init__(self,fname='',sname='', nframes = -1, fps = 30,codec='X264',sh='',sw=''):
        self.fname = fname
        self.sname = sname
        self.nframes = nframes
        self.codec=codec
        self.h = 0
        self.w = 0
        self.FPS = fps
        self.nc = 3
        self.i = 0

        try:
            if self.fname != '':
                self.vid = self.get_video()
                self.get_info()
                self.sh = 0
                self.sw = 0
            if self.sname != '':
                if sh=='' and sw=='':
                    self.sh = self.h
                    self.sw = self.w
                else:
                    self.sw=sw
                    self.sh=sh
                self.svid = self.create_video()

        except Exception as ex:
            from traceback import print_exc
            print_exc()

    def load_frame(self):
        try:
            frame = self._read_frame()
            self.i += 1
            return frame
        except Exception as ex:
            print('Error: %s', ex)

    def height(self):
        return self.h

    def width(self):
        return self.w

    def fps(self):
        return self.FPS

    def counter(self):
        return self.i

    def frame_count(self):
        return self.nframes

    def get_video(self):
        '''
        implement your own
        '''
        raise NotImplementedError()

    def get_info(self):
        '''
        implement your own
        '''
        raise NotImplementedError()

    def create_video(self):
        '''
        implement your own
        '''
        raise NotImplementedError()

    def _read_frame(self):
        '''
        implement your own
        '''
        raise NotImplementedError()

    def save_frame(self,frame):
        '''
        implement your own
        '''
        raise NotImplementedError()

    def close(self):
        '''
        implement your own
        '''
        raise NotImplementedError()


class VideoProcessorCV(VideoProcessor):
    '''
    OpenCV implementation of VideoProcessor
    requires opencv-python==3.4.0.12
    '''
    def __init__(self, *args, **kwargs):
        super(VideoProcessorCV, self).__init__(*args, **kwargs)

    def get_video(self):
         return cv2.VideoCapture(self.fname)

    def get_info(self):
        self.w = int(self.vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h = int(self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        all_frames = int(self.vid.get(cv2.CAP_PROP_FRAME_COUNT))
        self.FPS = self.vid.get(cv2.CAP_PROP_FPS)
        self.nc = 3
        if self.nframes == -1 or self.nframes>all_frames:
            self.nframes = all_frames
        print(self.nframes)

    def create_video(self):
        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        return cv2.VideoWriter(self.sname,fourcc, self.FPS, (self.sw,self.sh),True)

    def _read_frame(self): #return RGB (rather than BGR)!
        #return cv2.cvtColor(np.flip(self.vid.read()[1],2), cv2.COLOR_BGR2RGB)
        return np.flip(self.vid.read()[1],2)

    def save_frame(self,frame):
        self.svid.write(np.flip(frame,2))

    def close(self):
        self.svid.release()
        self.vid.release()

class GenericVideoProcessor(VideoProcessor):
    """
    using FramePicker and FrameWriter
    """
    @classmethod
    def prepare(cls, reader="opencv", writer="opencv"):
        readercls = frame_pickers.get_frame_picker_class(reader)
        writercls = frame_writers.get_frame_writer_class(writer)
        def init(*args, **kwargs):
            return cls(*args, readercls=readercls, writercls=writercls, **kwargs)
        return init

    def __init__(self,
                 fname="", # source path
                 sname="", # dest path
                 nframes=-1,
                 fps=None,
                 codec="X264", # in FOURCC
                 readercls=None,
                 writercls=None,
                 **kwargs):
        self._rcls = readercls
        self._wcls = writercls
        self._it   = None
        print(f"reader={self._rcls.__name__}, writer={self._wcls.__name__}")
        super().__init__(fname=fname,
                         sname=sname,
                         nframes=nframes,
                         fps=fps,
                         codec=codec)

    def get_video(self):
        print(f"opening source: {self.fname}")
        return self._rcls(self.fname)

    def get_info(self):
        self.w = self.vid.width
        self.h = self.vid.height
        if self.FPS is None:
            self.FPS = self.vid.fps
        self.nc = self.vid.nchan
        if self.nframes == -1 or self.nframes > self.vid.nframes:
            self.nframes = self.vid.nframes

    def create_video(self):
        print(f"opening sink: {self.sname}")
        return self._wcls(path=self.sname,
                          width_height=(self.sw, self.sh),
                          fps=self.FPS,
                          codec=self.codec)

    def _read_frame(self):
        if self._it is None:
            self._it = iter(self.vid.iter_frames(assert_color=True))
        return np.array(next(self._it), copy=True)

    def save_frame(self, frame):
        self.svid.write(frame)

    def close(self):
        self._it = None
        self.vid.close()
        self.svid.close()


def get_frame_processor_class(reader="opencv", writer="opencv"):
    if (reader == "opencv") and (writer == "opencv"):
        return VideoProcessorCV
    else:
        return GenericVideoProcessor.prepare(reader, writer)
