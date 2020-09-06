# Current issues

## `destfolder` not clear

`destfolder` in e.g. `analyze_video` seems to refer to where the analyzed `.h5` annotation files reside, but it is referred to as `destfolder` also in `create_labeled_video`, although the video is not created in that folder.

probably it is better to refer to it as `annotation_dir` in `create_labeled_video`, whereas it would be easier to have a proper `destfolder` keyword argument.

