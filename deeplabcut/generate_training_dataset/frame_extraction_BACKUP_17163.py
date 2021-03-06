"""
DeepLabCut2.0 Toolbox (deeplabcut.org)
© A. & M. Mathis Labs
https://github.com/AlexEMG/DeepLabCut

Please see AUTHORS for contributors.
https://github.com/AlexEMG/DeepLabCut/blob/master/AUTHORS
Licensed under GNU Lesser General Public License v3.0
"""

<<<<<<< HEAD
def extract_frames(config,
                    mode='automatic',
                    algo='kmeans',
                    crop=False,
                    userfeedback=True,
                    cluster_step=1,
                    cluster_resizewidth=30,
                    cluster_color=False,
                    driver='opencv',
                    slider_width=25):
=======

def select_cropping_area(config, videos=None):
    """
    Interactively select the cropping area of all videos in the config.
    A user interface pops up with a frame to select the cropping parameters.
    Use the left click to draw a box and hit the button 'set cropping parameters'
    to store the cropping parameters for a video in the config.yaml file.

    Parameters
    ----------
    config : string
        Full path of the config.yaml file as a string.

    videos : optional (default=None)
        List of videos whose cropping areas are to be defined. Note that full paths are required.
        By default, all videos in the config are successively loaded.

    Returns
    -------
    cfg : dict
        Updated project configuration
    """
    from deeplabcut.utils import auxiliaryfunctions, auxfun_videos

    cfg = auxiliaryfunctions.read_config(config)
    if videos is None:
        videos = cfg["video_sets"]

    for video in videos:
        coords = auxfun_videos.draw_bbox(video)
        if coords:
            cfg["video_sets"][video] = {
                "crop": ", ".join(
                    map(
                        str,
                        [
                            int(coords[0]),
                            int(coords[2]),
                            int(coords[1]),
                            int(coords[3]),
                        ],
                    )
                )
            }

    auxiliaryfunctions.write_config(config, cfg)
    return cfg


def extract_frames(
    config,
    mode="automatic",
    algo="kmeans",
    crop=False,
    userfeedback=True,
    cluster_step=1,
    cluster_resizewidth=30,
    cluster_color=False,
    opencv=True,
    slider_width=25,
):
>>>>>>> f0e9159048cec39cefdd1c10ca0aea08ccaf7394
    """
    Extracts frames from the videos in the config.yaml file. Only the videos in the config.yaml will be used to select the frames.\n
    Use the function ``add_new_video`` at any stage of the project to add new videos to the config file and extract their frames.

    The provided function either selects frames from the videos in a randomly and temporally uniformly distributed way (uniform), \n
    by clustering based on visual appearance (k-means), or by manual selection.

    Three important parameters for automatic extraction: numframes2pick, start and stop are set in the config file.

<<<<<<< HEAD
    Please refer to the user guide for more details on methods and parameters https://www.biorxiv.org/content/biorxiv/early/2018/11/24/476531.full.pdf
=======
    Please refer to the user guide for more details on methods and parameters https://www.nature.com/articles/s41596-019-0176-0
    or the preprint: https://www.biorxiv.org/content/biorxiv/early/2018/11/24/476531.full.pdf
>>>>>>> f0e9159048cec39cefdd1c10ca0aea08ccaf7394

    Parameters
    ----------
    config : string
        Full path of the config.yaml file as a string.

    mode : string
        String containing the mode of extraction. It must be either ``automatic`` or ``manual``.

    algo : string
        String specifying the algorithm to use for selecting the frames. Currently, deeplabcut supports either ``kmeans`` or ``uniform`` based selection. This flag is
        only required for ``automatic`` mode and the default is ``uniform``. For uniform, frames are picked in temporally uniform way, kmeans performs clustering on downsampled frames (see user guide for details).
        Note: color information is discarded for kmeans, thus e.g. for camouflaged octopus clustering one might want to change this.

    crop : bool, optional
<<<<<<< HEAD
        If this is set to True, a user interface pops up with a frame to select the cropping parameters. Use the left click to draw a cropping area and hit the button set cropping parameters to save the cropping parameters for a video.
        The default is ``False``; if provided it must be either ``True`` or ``False``.
=======
        If True, video frames are cropped according to the corresponding coordinates stored in the config.yaml.
        Alternatively, if cropping coordinates are not known yet, crop='GUI' triggers a user interface
        where the cropping area can be manually drawn and saved.
>>>>>>> f0e9159048cec39cefdd1c10ca0aea08ccaf7394

    userfeedback: bool, optional
        If this is set to false during automatic mode then frames for all videos are extracted. The user can set this to true, which will result in a dialog,
        where the user is asked for each video if (additional/any) frames from this video should be extracted. Use this, e.g. if you have already labeled
        some folders and want to extract data for new videos.

    cluster_resizewidth: number, default: 30
        For k-means one can change the width to which the images are downsampled (aspect ratio is fixed).

    cluster_step: number, default: 1
        By default each frame is used for clustering, but for long videos one could only use every nth frame (set by: cluster_step). This saves memory before clustering can start, however,
        reading the individual frames takes longer due to the skipping.

    cluster_color: bool, default: False
        If false then each downsampled image is treated as a grayscale vector (discarding color information). If true, then the color channels are considered. This increases
        the computational complexity.

<<<<<<< HEAD
    driver: str, default: 'opencv'
        driver for loading & extracting: 'opencv' (default), 'moviepy' (legacy) or 'skvideo' (experimental)
=======
    opencv: bool, default: True
        Uses openCV for loading & extractiong (otherwise moviepy (legacy))
>>>>>>> f0e9159048cec39cefdd1c10ca0aea08ccaf7394

    slider_width: number, default: 25
        Width of the video frames slider, in percent of window

    Examples
    --------
    for selecting frames automatically with 'kmeans' and want to crop the frames.
    >>> deeplabcut.extract_frames('/analysis/project/reaching-task/config.yaml','automatic','kmeans',True)
    --------
    for selecting frames automatically with 'kmeans' and defining the cropping area at runtime.
    >>> deeplabcut.extract_frames('/analysis/project/reaching-task/config.yaml','automatic','kmeans','GUI')
    --------
    for selecting frames automatically with 'kmeans' and considering the color information.
    >>> deeplabcut.extract_frames('/analysis/project/reaching-task/config.yaml','automatic','kmeans',cluster_color=True)
    --------
    for selecting frames automatically with 'uniform' and want to crop the frames.
    >>> deeplabcut.extract_frames('/analysis/project/reaching-task/config.yaml','automatic',crop=True)
    --------
    for selecting frames manually,
    >>> deeplabcut.extract_frames('/analysis/project/reaching-task/config.yaml','manual')
    --------
    for selecting frames manually, with a 60% wide frames slider
    >>> deeplabcut.extract_frames('/analysis/project/reaching-task/config.yaml','manual', slider_width=60)

    While selecting the frames manually, you do not need to specify the ``crop`` parameter in the command. Rather, you will get a prompt in the graphic user interface to choose
    if you need to crop or not.
    --------

    """
    import os
    import sys
    import numpy as np
    from pathlib import Path
    from skimage import io
    from skimage.util import img_as_ubyte
    from deeplabcut.utils import frameselectiontools
    from deeplabcut.utils import auxiliaryfunctions
<<<<<<< HEAD
    from deeplabcut.utils import frame_pickers
    from matplotlib.widgets import RectangleSelector
=======
>>>>>>> f0e9159048cec39cefdd1c10ca0aea08ccaf7394

    if mode == "manual":
        wd = Path(config).resolve().parents[0]
        os.chdir(str(wd))
        from deeplabcut.generate_training_dataset import frame_extraction_toolbox
        from deeplabcut.utils import select_crop_parameters

        frame_extraction_toolbox.show(config, slider_width)

    elif mode == "automatic":
        config_file = Path(config).resolve()
        cfg = auxiliaryfunctions.read_config(config_file)
        print("Config file read successfully.")

<<<<<<< HEAD
        numframes2pick = cfg['numframes2pick']
        start = cfg['start']
        stop = cfg['stop']

        # Check for variable correctness
        if start>1 or stop>1 or start<0 or stop<0 or start>=stop:
            raise Exception("Erroneous start or stop values. Please correct it in the config file.")
        if numframes2pick<1 and not int(numframes2pick):
            raise Exception("Perhaps consider extracting more, or a natural number of frames.")

        videos       = cfg['video_sets'].keys()
        picker_class = frame_pickers.get_frame_picker_class(driver=driver)
        for vindex,video in enumerate(videos):
            #plt.close("all")
            global coords
            coords = cfg['video_sets'][video]['crop'].split(',')

=======
        numframes2pick = cfg["numframes2pick"]
        start = cfg["start"]
        stop = cfg["stop"]

        # Check for variable correctness
        if start > 1 or stop > 1 or start < 0 or stop < 0 or start >= stop:
            raise Exception(
                "Erroneous start or stop values. Please correct it in the config file."
            )
        if numframes2pick < 1 and not int(numframes2pick):
            raise Exception(
                "Perhaps consider extracting more, or a natural number of frames."
            )

        videos = cfg["video_sets"].keys()
        if opencv:
            import cv2
        else:
            from moviepy.editor import VideoFileClip

        has_failed = []
        for vindex, video in enumerate(videos):
>>>>>>> f0e9159048cec39cefdd1c10ca0aea08ccaf7394
            if userfeedback:
                print(
                    "Do you want to extract (perhaps additional) frames for video:",
                    video,
                    "?",
                )
                askuser = input("yes/no")
            else:
<<<<<<< HEAD
                askuser="yes"

            if askuser=='y' or askuser=='yes' or askuser=='Ja' or askuser=='ha': # multilanguage support :)
                picker = picker_class(video)
                if crop==True:
                    from deeplabcut.utils import select_crop_parameters
                    image = picker.pick_at_fraction(start)

                    fname = Path(video)
                    output_path = Path(config).parents[0] / 'labeled-data' / fname.stem

                    if output_path.exists() :
                        fig,ax = plt.subplots(1)
                        # Call the GUI to select the cropping parameters
                        coords = select_crop_parameters.show(config,image)
                        # Update the config.yaml file with current cropping parameters
                        cfg['video_sets'][video] = {'crop': ', '.join(map(str, [int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])]))}
                        auxiliaryfunctions.write_config(config_file,cfg)

                        def __save_random(picker, start, stop):
                            """a full frame from random location (good for augmentation)"""
                            index = int(start*(picker.nframes) + np.random.rand()*(picker.nframes)*(stop-start))
                            picker.save_single(index, output_dir=output_path)

                        if len(os.listdir(output_path))==0: #check if empty
                                #store full frame from random location (good for augmentation)
                                __save_random(picker, start, stop)

                        else:
                              askuser=input ("The directory already contains some frames. Do you want to add to it?(yes/no): ")
                              if askuser=='y' or askuser=='yes' or askuser=='Y' or askuser=='Yes':
                                  __save_random(picker, start, stop)
                              else:
                                  sys.exit("Delete the frames and try again later!")

                    picker.crop(coords)

                else:
                    numframes2pick=cfg['numframes2pick']+1 # without cropping a full size frame will not be extracted >> thus one more frame should be selected in next stage.

                print("Extracting frames based on %s ..." %algo)
                frames2pick = frameselectiontools.run(picker, numframes2pick-1, start=start, stop=stop,
                                        step=cluster_step, resizewidth=cluster_resizewidth)

                output_dir = Path(config).parents[0] / 'labeled-data' / Path(video).stem
                picker.save_multiple(frames2pick, output_dir=output_dir, basename='img')
                picker.close()
    else:
        print("Invalid MODE. Choose either 'manual' or 'automatic'. Check ``help(deeplabcut.extract_frames)`` on python and ``deeplabcut.extract_frames?`` \
              for ipython/jupyter notebook for more details.")

    print("\nFrames were selected.\nYou can now label the frames using the function 'label_frames' (if you extracted enough frames for all videos).")

def line_select_callback(eclick, erelease):
        'eclick and erelease are the press and release events'
        global coords
        new_x1, new_y1 = eclick.xdata, eclick.ydata
        new_x2, new_y2 = erelease.xdata, erelease.ydata
        coords = [str(int(new_x1)),str(int(new_x2)),str(int(new_y1)),str(int(new_y2))]
=======
                askuser = "yes"

            if (
                askuser == "y"
                or askuser == "yes"
                or askuser == "Ja"
                or askuser == "ha"
                or askuser == "oui"
                or askuser == "ouais"
            ):  # multilanguage support :)
                if opencv:
                    cap = cv2.VideoCapture(video)
                    fps = cap.get(
                        5
                    )  # https://docs.opencv.org/2.4/modules/highgui/doc/reading_and_writing_images_and_video.html#videocapture-get
                    nframes = int(cap.get(7))
                else:
                    # Moviepy:
                    clip = VideoFileClip(video)
                    fps = clip.fps
                    nframes = int(np.ceil(clip.duration * 1.0 / fps))
                if not nframes:
                    print("Video could not be opened. Skipping...")
                    continue

                indexlength = int(np.ceil(np.log10(nframes)))

                fname = Path(video)
                output_path = Path(config).parents[0] / "labeled-data" / fname.stem

                if output_path.exists():
                    if len(os.listdir(output_path)):
                        if userfeedback:
                            askuser = input(
                                "The directory already contains some frames. Do you want to add to it?(yes/no): "
                            )
                        if not (
                            askuser == "y"
                            or askuser == "yes"
                            or askuser == "Y"
                            or askuser == "Yes"
                        ):
                            sys.exit("Delete the frames and try again later!")

                if crop == "GUI":
                    cfg = select_cropping_area(config, [video])
                coords = cfg["video_sets"][video]["crop"].split(",")
                if crop and not opencv:
                    clip = clip.crop(
                        y1=int(coords[2]),
                        y2=int(coords[3]),
                        x1=int(coords[0]),
                        x2=int(coords[1]),
                    )
                elif not crop:
                    coords = None

                print("Extracting frames based on %s ..." % algo)
                if algo == "uniform":
                    if opencv:
                        frames2pick = frameselectiontools.UniformFramescv2(
                            cap, numframes2pick, start, stop
                        )
                    else:
                        frames2pick = frameselectiontools.UniformFrames(
                            clip, numframes2pick, start, stop
                        )
                elif algo == "kmeans":
                    if opencv:
                        frames2pick = frameselectiontools.KmeansbasedFrameselectioncv2(
                            cap,
                            numframes2pick,
                            start,
                            stop,
                            crop,
                            coords,
                            step=cluster_step,
                            resizewidth=cluster_resizewidth,
                            color=cluster_color,
                        )
                    else:
                        frames2pick = frameselectiontools.KmeansbasedFrameselection(
                            clip,
                            numframes2pick,
                            start,
                            stop,
                            step=cluster_step,
                            resizewidth=cluster_resizewidth,
                            color=cluster_color,
                        )
                else:
                    print(
                        "Please implement this method yourself and send us a pull request! Otherwise, choose 'uniform' or 'kmeans'."
                    )
                    frames2pick = []

                if not len(frames2pick):
                    print("Frame selection failed...")
                    return

                output_path = (
                    Path(config).parents[0] / "labeled-data" / Path(video).stem
                )
                is_valid = []
                if opencv:
                    for index in frames2pick:
                        cap.set(1, index)  # extract a particular frame
                        ret, frame = cap.read()
                        if ret:
                            image = img_as_ubyte(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                            img_name = (
                                str(output_path)
                                + "/img"
                                + str(index).zfill(indexlength)
                                + ".png"
                            )
                            if crop:
                                io.imsave(
                                    img_name,
                                    image[
                                        int(coords[2]) : int(coords[3]),
                                        int(coords[0]) : int(coords[1]),
                                        :,
                                    ],
                                )  # y1 = int(coords[2]),y2 = int(coords[3]),x1 = int(coords[0]), x2 = int(coords[1]
                            else:
                                io.imsave(img_name, image)
                            is_valid.append(True)
                        else:
                            print("Frame", index, " not found!")
                            is_valid.append(False)
                    cap.release()
                else:
                    for index in frames2pick:
                        try:
                            image = img_as_ubyte(clip.get_frame(index * 1.0 / clip.fps))
                            img_name = (
                                str(output_path)
                                + "/img"
                                + str(index).zfill(indexlength)
                                + ".png"
                            )
                            io.imsave(img_name, image)
                            if np.var(image) == 0:  # constant image
                                print(
                                    "Seems like black/constant images are extracted from your video. Perhaps consider using opencv under the hood, by setting: opencv=True"
                                )
                            is_valid.append(True)
                        except FileNotFoundError:
                            print("Frame # ", index, " does not exist.")
                            is_valid.append(False)
                    clip.close()
                    del clip

                if not any(is_valid):
                    has_failed.append(True)
                else:
                    has_failed.append(False)

        if all(has_failed):
            print("Frame extraction failed. Video files must be corrupted.")
            return
        elif any(has_failed):
            print("Although most frames were extracted, some were invalid.")
        else:
            print("Frames were successfully extracted.")
        print(
            "\nYou can now label the frames using the function 'label_frames' "
            "(if you extracted enough frames for all videos)."
        )
    else:
        print(
            "Invalid MODE. Choose either 'manual' or 'automatic'. Check ``help(deeplabcut.extract_frames)`` on python and ``deeplabcut.extract_frames?`` \
              for ipython/jupyter notebook for more details."
        )
>>>>>>> f0e9159048cec39cefdd1c10ca0aea08ccaf7394
