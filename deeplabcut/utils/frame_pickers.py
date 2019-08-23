"""
DeepLabCut2.0 Toolbox (deeplabcut.org)
© A. & M. Mathis Labs
https://github.com/AlexEMG/DeepLabCut

Please see AUTHORS for contributors.
https://github.com/AlexEMG/DeepLabCut/blob/master/AUTHORS
Licensed under GNU Lesser General Public License v3.0
"""
import numpy as np
import os

class FramePicker(object):
    """the base driver class for extracting frames from a video."""
    def __init__(self, path):
        from skimage import io
        from skimage.util import img_as_ubyte

        self.path         = path
        self.nframes      = 0
        self.img_as_ubyte = img_as_ubyte
        self.crop         = None

    def __getattr__(self, name):
        if name == 'indexlength':
            if not hasattr(self, '_indexlength'):
                self._indexlength = int(np.ceil(np.log10(self.nframes)))
            return self._indexlength
        elif name in ('is_colored', 'ncolors'):
            if not hasattr(self, '_is_colored'):
                self._ncolors    = self.get_ncolorchannels()
                self._is_colored = self._ncolors > 1
            return getattr(self, '_' + name)
        else:
            return super(FramePicker, self).__getattr__(name)

    def get_ncolorchannels(self):
        """returns the number of color channels. returning zero means that the frame is 2D."""
        raise NotImplementedError()

    def set_crop(self, coords):
        self.crop = coords

    def set_resize(self, resizewidth):
        raise NotImplementedError()

    def pick_single(self, index):
        raise NotImplementedError()

    def pick_multiple(self, indices):
        raise NotImplementedError()

    def pick_at_fraction(self, frac):
        raise NotImplementedError()

    def save_impl(self, index, image=None, output_dir='', basename='img'):
        if image is None:
            image = self.pick_single(index)
        savepath = os.path.join(output_dir, "{base}{index}.png".format(
                                base=basename, index=str(index).zfill(self.indexlength)))
        io.imsave(savepath, image)

    def save_single(self, index, output_dir='', basename='img'):
        self.save_impl(index, image=None, output_dir=output_dir, basename=basename)

class OpenCVPicker(FramePicker):
    """the default frame picker based on OpenCV."""
    def __init__(self, path):
        import cv2
        super(OpenCVPicker, self).__init__(path)
        self.cap      = cv2.VideoCapture(path)
        self.fps      = self.cap.get(cv2.CAP_PROP_FPS)
        self.width    = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height   = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.nframes  = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.nframes*1./(self.fps)
        self.offset   = 0
        self.resize   = None

    def set_resize(self, resizewidth):
        ratio  = resizewidth*1. / self.width
        if ratio > 1:
            raise ValueError("Choice of resizewidth actually upsamples!")
        self.resize = ratio

    def pick_single(self, index, crop=False, resize=False, transform_color=True):
        if self.offset != index:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ret, img = self.cap.read()
        self.offset += 1
        if not ret:
            raise RuntimeError("could not read from the specified position")
        if crop and self.crop:
            img = img[int(self.crop[2]):int(self.crop[3]),int(self.crop[0]):int(self.crop[1])]
        if resize and self.resize:
            img = cv2.resize(img, None, fx=self.resize, fy=self.resize,
                             interpolation=cv2.INTER_NEAREST)
        if transform_color
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return self.img_as_ubyte(img)

    def pick_at_fraction(self, frac):
        self.cap.set(cv2.CAP_PROP_POS_MSEC, frac*self.duration*1000)
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("could not read from the specified position")
        return self.img_as_ubyte(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

class MoviePyPicker(FramePicker):
    """the legacy frame picker based on MoviePy."""
    def __init__(self, path):
        from moviepy.editor import VideoFileClip
        super(MoviePyPicker, self).__init__(path)
        self.clip     = VideoFileClip(video)
        self.fps      = self.clip.fps
        self.duration = self.clip.duration
        self.nframes  = int(np.ceil(self.clip.duration*1./(self.fps)))
        self.resized  = None
        self.ncolors  = None

    def get_ncolorchannels(self):
        frame0 = self.img_as_ubyte(self.clip.get_frame(0))
        if np.ndim(frame0) == 3:
            return frame0.shape[2]
        else:
            return 0

    def set_crop(self, coords):
        self.clip = self.clip.crop(y1 = int(coords[2]),
                                   y2 = int(coords[3]),
                                   x1 = int(coords[0]),
                                   x2 = int(coords[1]))

    def set_resize(self, resizewidth):
        self.resized = self.clip.resize(width=resizewidth)

    def pick_single(self, index, crop=None, resize=False, transform_color=None):
        """`crop` and `transform_color` are not taken into account."""
        if resize and self.resized:
            return self.img_as_ubyte(self.resized.get_frame(index * 1. / (self.fps)))
        else:
            return self.img_as_ubyte(self.clip.get_frame(index * 1. / (self.fps)))

    def pick_at_fraction(self, frac):
        return self.img_as_ubyte(self.clip.get_frame(frac*self.duration)) #frame is accessed by index *1./clip.fps (fps cancels)

class SkVideoPicker(FramePicker):
    """an experimental, rather slow frame picker based on scikit-video/ffmpeg."""
    def __init__(self, path):
        from skvideo.io import FFmpegReader
        import cv2
        super(SkVideoPicker, self).__init__(path)
        self.reader   = FFmpegReader(path)
        self.nframes, self.width, self.height, self.nchan = self._getinfo()
        self.duration = self.nframes
        self.iterator = None
        self.offset   = 0

    def get_ncolorchannels(self):
        return self.nchan

    def _getinfo(self):
        n = 0
        for image in self.reader.nextFrame():
            n += 1
        width, height, nchan = image.shape # bleeds through
        return n, width, height, nchan

    def set_resize(self, resizewidth):
        ratio  = resizewidth*1. / self.width
        if ratio > 1:
            raise ValueError("Choice of resizewidth actually upsamples!")
        self.resize = ratio

    def rewind(self):
        self.iterator = iter(self.reader.nextFrame())
        self.offset   = 0

    def pick_single(self, index, crop=False, resize=False, transform_color=None):
        if index >= self.nframes:
            raise ValueError("index out of range: {}".format(index))
        if (self.iterator is None) or (self.offset > index):
            self.rewind()
        while self.offset < index:
            img = next(self.iterator)
            self.offset += 1

        if crop and self.crop:
            img = img[int(self.crop[2]):int(self.crop[3]),int(self.crop[0]):int(self.crop[1])]
        if resize and self.resize:
            img = cv2.resize(img, None, fx=self.resize, fy=self.resize,
                             interpolation=cv2.INTER_NEAREST)
        img = self.img_as_ubyte(img)

    def pick_at_fraction(self, frac):
        index = int(frac*self.nframes)
        return self.pick_single(index)

def get_frame_picker_class(driver='opencv'):
    if driver == 'moviepy':
        return MoviePyPicker
    elif driver == 'skvideo':
        return SkVideoPicker
    else:
        return OpenCVPicker
