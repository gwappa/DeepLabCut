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
from traceback import print_exc as _print_exc

from skvideo.io import FFmpegReader
from skimage import io
from skimage.util import img_as_ubyte
from deeplabcut.utils.imageop import imresize
import cv2

class FramePicker(object):
    """the base driver class for extracting frames from a video."""
    driver = None

    def __init__(self, path):
        self.path         = str(path)
        self.nframes      = 0
        self.img_as_ubyte = img_as_ubyte
        self.crop         = None
        self.resize       = None
        self.is_open      = False

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
            raise AttributeError(name)

    def close(self):
        raise NotImplementedError()

    def condition_image_impl(self, img, crop=False, resize=False, transform_color=False, assert_color=False):
        if (crop == True) and (self.crop is not None):
            img = img[int(self.crop[2]):int(self.crop[3]),int(self.crop[0]):int(self.crop[1])]
        if (resize == True) and (self.resize is not None):
            img = cv2.resize(img, None, fx=self.resize, fy=self.resize,
                             interpolation=cv2.INTER_NEAREST)
        if assert_color == True:
            if img.ndim == 2:
                img = np.stack([img, img, img], axis=-1)
        if transform_color == True:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def iter_frames(self, crop=False, resize=False, transform_color=True, assert_color=False):
        raise NotImplementedError()

    def get_ncolorchannels(self):
        """returns the number of color channels. returning zero means that the frame is 2D."""
        raise NotImplementedError()

    def set_crop(self, coords):
        self.crop = coords

    def set_resize(self, resizewidth):
        raise NotImplementedError()

    def pick_single(self, index, crop=False, resize=False, transform_color=True):
        raise NotImplementedError()

    def pick_at_fraction(self, frac):
        raise NotImplementedError()

    def save_impl(self, index, image=None, output_dir='', basename='img',
                    crop=False, resize=False, transform_color=True, indexwidth=5):
        if image is None:
            image = self.pick_single(index)
        savepath = os.path.join(str(output_dir) , "{base}{index}.png".format(
                                base=basename, index=str(index).zfill(indexwidth)))
        io.imsave(savepath, image)

    def save_single(self, index, output_dir='', basename='img',
                    crop=False, resize=False, transform_color=True, indexwidth=5):
        self.save_impl(index, image=None, output_dir=output_dir, basename=basename,
                    crop=crop, resize=resize, transform_color=transform_color, indexwidth=indexwidth)

    def save_multiple(self, indices, output_dir='', basename='img',
                    crop=False, resize=False, transform_color=True):
        for index in sorted(indices):
            try:
                self.save_impl(index, image=None, output_dir=output_dir, basename=basename,
                            crop=crop, resize=resize, transform_color=transform_color)
            except:
                _print_exc()

class OpenCVPicker(FramePicker):
    """the default frame picker based on OpenCV."""
    driver = "opencv"

    def __init__(self, path):
        super(OpenCVPicker, self).__init__(path)
        self.cap      = cv2.VideoCapture(str(path))
        self.is_open  = True
        self.fps      = self.cap.get(cv2.CAP_PROP_FPS)
        self.width    = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height   = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.nframes  = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.nframes*1./(self.fps)
        self.offset   = 0
        self.resize   = None

    def close(self):
        if self.is_open == True:
            self.cap.release()
            del self.cap
            self.is_open = False

    def get_ncolorchannels(self):
        ret, img = self.cap.read()
        self.offset += 1
        if not ret:
            raise RuntimeError("could not read from the stream")
        if np.ndim(img) == 2:
            return 0
        else:
            return np.shape(img)[2]

    def iter_frames(self, crop=False, resize=False, transform_color=True):
        if self.offset != 0:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        while self.cap.is_open():
            try:
                yield self.read(crop=crop, resize=resize, transform_color=transform_color)
            except RuntimeError:
                break

    def set_resize(self, resizewidth):
        ratio  = resizewidth*1. / self.width
        if ratio > 1:
            raise ValueError("Choice of resizewidth actually upsamples!")
        self.resize = ratio

    def read(self, crop=False, resize=False, transform_color=True):
        ret, img = self.cap.read()
        self.offset += 1
        if not ret:
            raise RuntimeError("could not read from the specified position")
        img = self.condition_image_impl(img, crop, resize, transform_color)
        return img_as_ubyte(img)

    def pick_single(self, index, crop=False, resize=False, transform_color=True):
        if self.offset != index:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        return self.read(crop=crop, resize=resize, transform_color=transform_color)

    def pick_at_fraction(self, frac):
        self.cap.set(cv2.CAP_PROP_POS_MSEC, frac*self.duration*1000)
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("could not read from the specified position")
        return self.img_as_ubyte(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

class MoviePyPicker(FramePicker):
    """the legacy frame picker based on MoviePy."""
    driver = "moviepy"

    def __init__(self, path):
        from moviepy.editor import VideoFileClip
        super(MoviePyPicker, self).__init__(path)
        self.clip     = VideoFileClip(str(path))
        self.is_open  = True
        self.fps      = self.clip.fps
        self.duration = self.clip.duration
        self.nframes  = int(np.ceil(self.clip.duration*1./(self.fps)))
        self.resized  = None
        self.ncolors  = None
        self.width    = np.nan
        self.height   = np.nan

    def close(self):
        if self.is_open == True:
            self.clip.close()
            del self.clip
            self.is_open = False

    def get_ncolorchannels(self):
        frame0 = self.img_as_ubyte(self.clip.get_frame(0))
        if np.ndim(frame0) == 3:
            return frame0.shape[2]
        else:
            return 0

    def iter_frames(self):
        # FIXME
        raise NotImplementedError()

    def set_crop(self, coords):
        self.clip = self.clip.crop(y1 = int(coords[2]),
                                   y2 = int(coords[3]),
                                   x1 = int(coords[0]),
                                   x2 = int(coords[1]))

    def set_resize(self, resizewidth):
        self.resized = self.clip.resize(width=resizewidth)

    def pick_single(self, index, crop=None, resize=False, transform_color=None):
        """`crop` and `transform_color` are not taken into account."""
        if (resize == True) and (self.resized is not None):
            return img_as_ubyte(self.resized.get_frame(index * 1. / (self.fps)))
        else:
            return img_as_ubyte(self.clip.get_frame(index * 1. / (self.fps)))

    def pick_at_fraction(self, frac):
        return img_as_ubyte(self.clip.get_frame(frac*self.duration)) #frame is accessed by index *1./clip.fps (fps cancels)

class SkVideoPicker(FramePicker):
    """an experimental, rather slow frame picker based on scikit-video/ffmpeg."""
    driver = "skvideo"

    def __init__(self, path):
        super(SkVideoPicker, self).__init__(path)
        self.reader   = FFmpegReader(str(path))
        self.is_open  = True
        self.nframes, self.width, self.height, self.nchan = self._getinfo()
        self.duration = self.nframes
        self.fps      = 1
        self.iterator = None
        self.offset   = 0

    def close(self):
        if self.is_open == True:
            self.reader.close()
            del self.reader
            self.is_open = False

    def get_ncolorchannels(self):
        return self.nchan

    def _getinfo(self):
        n = 0
        for image in self.reader.nextFrame():
            n += 1
        if image.ndim == 2:
            height, width = image.shape
            nchan = 0
        else:
            height, width, nchan = image.shape[:3]
        self.rewind()
        return n, width, height, nchan

    def set_resize(self, resizewidth):
        ratio  = resizewidth*1. / self.width
        if ratio > 1:
            raise ValueError("Choice of resizewidth actually upsamples!")
        self.resize = ratio

    def iter_frames(self, crop=False, resize=False, transform_color=True):
        if (self.iterator is None) or (self.offset > 0):
            self.rewind()
        try:
            while True:
                yield self.read(None, crop=crop, resize=resize, transform_color=transform_color)
        except StopIteration:
            pass

    def rewind(self):
        self.reader.close()
        self.reader   = FFmpegReader(self.path)
        self.iterator = iter(self.reader.nextFrame())
        self.offset   = 0

    def seek(self, index):
        if index >= self.nframes:
            raise ValueError("index out of range: {}".format(index))
        if (self.iterator is None) or (self.offset > index):
            self.rewind()

        while True:
            try:
                img = next(self.iterator)
                self.offset += 1
                if self.offset == (index+1):
                    return img
            except StopIteration:
                raise RuntimeError("reached end of stream while seeking")

    def read(self, img=None, crop=False, resize=False, transform_color=None):
        if img is None:
            img = next(self.iterator)
        if (crop == True) and (self.crop is not None):
            img = img[int(self.crop[2]):int(self.crop[3]),int(self.crop[0]):int(self.crop[1])]
        if (resize == True) and (self.resize is not None):
            img = imresize(img, self.resize, interp='nearest')
        return img_as_ubyte(img)

    def pick_single(self, index, crop=False, resize=False, transform_color=None):
        img = self.seek(index)
        return self.read(img, crop=crop, resize=resize, transform_color=transform_color)

    def pick_at_fraction(self, frac):
        index = int(frac*self.nframes)
        return self.pick_single(index)

class NumpyPicker(FramePicker):
    driver = "numpy"

    def __init__(self, path):
        super().__init__(path)
        loaded = np.load(str(path))
        try:
            self._data = loaded["frames"]
            dt = np.diff(loaded["timestamps"]).mean()
        except:
            self._data = loaded
        self.nframes, self.height, self.width = self._data.shape[:3]
        if self._data.ndim == 4:
            self.nchan = self._data.shape[3]
        else:
            self.nchan = 1
        self.dt  = dt
        self.fps = 1 / dt
        self.duration = self.nframes * self.dt

    def close(self):
        pass

    def iter_frames(self, crop=False, resize=False, transform_color=False, assert_color=False):
        for i in range(self.nframes):
            yield self.pick_single(i, crop, resize, transform_color, assert_color)

    def get_ncolorchannels(self):
        ndim  = self._data.ndim
        shape = self._data.shape
        if ndim == 4:
            return shape[-1]
        elif ndim == 3:
            return 0
        else:
            raise ValueError(f"expected 3-d or 4-d data, got {ndim}-d")

    def set_resize(self, resizewidth):
        self._resizewidth = resizewidth

    def pick_single(self, index,
                    crop=False,
                    resize=False,
                    transform_color=False,
                    assert_color=False):
        # TODO: no need to use img_as_ubyte?
        img = self.img_as_ubyte(self._data[index])
        return self.condition_image_impl(img,
                                         crop,
                                         resize,
                                         transform_color=False,
                                         assert_color=assert_color)

    def pick_at_fraction(self, frac):
        index = int(self.nframes * frac)
        return self.pick_single(index)

def get_frame_picker_class(driver='opencv'):
    if driver == 'moviepy':
        return MoviePyPicker
    elif driver == 'skvideo':
        return SkVideoPicker
    elif driver == 'numpy':
        return NumpyPicker
    else:
        return OpenCVPicker

def get_frame_picker(path, driver='opencv'):
    return get_frame_picker_class(driver)(path)
