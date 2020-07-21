"""
classes for writing frames.
"""
from pathlib import Path
import numpy as np
import cv2

writers = {}

class FrameWriter(object):
    def __init__(self, path, width_height, fps=30, *args, **kwargs):
        self.path    = path
        self.width, self.height = width_height
        self.fps     = fps if fps > 0 else 1
        self._open(*args, **kwargs)
        print(f"opened a writer to: {self.path}")

    def _open(self, *args, **kwargs):
        raise NotImplementedError()

    def _close(self):
        raise NotImplementedError()

    def _write(self):
        raise NotImplementedError()

    def close(self):
        self._close()
        print(f"closed: {self.path}")

    def write(self, frame):
        self._write(frame)
        
try:
    import cv2
    class CV2Writer(FrameWriter):
        def __init__(self, path, width_height, fps=30, codec="mp4v"):
            super().__init__(path, width_height, fps, fourcc=codec)

        def _open(self, fourcc="mp4v", **kwargs):
            self._out = cv2.VideoWriter(str(self.path),
                                        cv2.VideoWriter_fourcc(fourcc),
                                        self.fps,
                                        (self.width, self.height),
                                        True)

        def _write(self, frame):
            self._out.write(np.flip(frame, 2))

        def _close(self):
            self._out.release()

    writers["opencv"] = CV2Writer
except ImportError:
    pass

try:
    from skvideo.io import FFmpegWriter
    class SkVideoWriter(FrameWriter):
        def __init__(self, path, width_height, inputdict=None, outputdict=None, *args, **kwargs):
            super().__init__(path,
                             width_height,
                             inputdict=inputdict,
                             outputdict=outputdict)

        def _open(self, inputdict=None, outputdict=None):
            inputdict  = {} if inputdict is None else dict(**inputdict)
            outputdict = {} if outputdict is None else dict(**outputdict)
            if all((key not in outputdict.keys()) for key in ("-r", "-framerate")):
                outputdict["-framerate"] = str(self.fps)
            self._out = FFmpegWriter(str(self.path),
                                    inputdict=inputdict,
                                    outputdict=outputdict)

        def _close(self):
            self._out.close()

        def _write(self, frame):
            self._out.writeFrame(frame)

    writers["skvideo"] = SkVideoWriter
except ImportError:
    pass

class NumpyWriter(FrameWriter):
    def __init__(self, path, width_height, *args, **kwargs):
        super().__init__(Path(path),
                         width_height)

    def _open(self):
        self._buf = []

    def _write(self, frame):
        self._buf.append(frame)

    def _close(self):
        size = len(self._buf)
        if size == 0:
            return
        if not self.path.parent.exists():
            self.path.parent.mkdir(parents=True)
        self._buf  = np.stack(self._buf, axis=0)
        timestamps = np.arange(size) / self.fps
        np.savez(str(self.path), time=timestamps, frames=self._buf)
        self._buf = None

writers["numpy"] = NumpyWriter

def get_frame_writer_class(driver="opencv"):
    return writers.get(driver, CV2Writer)

def get_frame_writer(path, width_height, driver="opencv", fps=30, *args, **kwargs):
    return get_frame_writer_class(driver)(path, width_height, fps, *args, **kwargs)
