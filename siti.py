#!/usr/bin/env python3

import argparse
import numpy as np
import pandas as pd
import skvideo.io
from scipy import ndimage
import matplotlib.pyplot as plt


def siti(filename: str, output: str = '', size: str = '0x0', pix_fmt: str = 'yuv420p', form: str = None,
         num_frames: int = 0, inputdict: dict = None) -> bool:
    """
    Python script to calculate SI/TI according recommendation ITU-T P.910.
    :param filename: Filename of video.
    :param output: CSV name to salve.
    :param size: Video dimension in "MxN" pixels.
    :param pix_fmt: Force pixel format to ffmpeg style.
    :param form: Force video format to ffmpeg style.
    :param num_frames: Number of frames to process.
    :param inputdict: A dict to ffmpeg backend.
    :return: None
    """
    print(f'Calculating SI/TI for {params["video"]}.')
    frame_counter = 0
    [width, height] = list(map(int, size.split('x')))
    if output in '':
        output = '.'.join(filename.split('.')[:-1]) + '.csv'  # Remove extension
    if form not in '':
        inputdict = {"-s": f"{width}x{height}", "-pix_fmt": pix_fmt, '-f': form}

    measures = {'SI': Si(), 'TI': Ti()}

    video = skvideo.io.vreader(fname=filename, inputdict=inputdict, as_grey=True, num_frames=num_frames)

    for frame in video:
        print(f"\nframe {frame_counter} of video {video}")
        frame_counter += 1
        frame = frame.reshape((height, width)).astype('float64')

        for measure in measures:
            value = measures[measure].calc(frame)
            print(f"{measure} -> {value}")

    df = pd.DataFrame()
    df["frame"] = range(1, measures['SI'].frame_counter + 1)
    df['SI'] = measures['SI'].values
    df['TI'] = measures['TI'].values
    df.to_csv(output, index=False)

    return True


class Features:
    def __init__(self):
        self.values = []
        self.frame_counter = 0

    def calc(self, frame):
        raise NotImplementedError("not implemented")


class Si(Features):
    def calc(self, frame):
        self.frame_counter += 1
        sobx = ndimage.sobel(frame, axis=0)
        soby = ndimage.sobel(frame, axis=1)
        sob = np.hypot(sobx, soby)
        sob_std = sob.std()
        si = round(sob_std, 4)
        self.values.append(si)
        # plt.imshow(frame, cmap='gray');plt.show()
        # plt.imshow(sob, cmap='gray');plt.show()
        # plt.savefig('temp\\frame' + str(self.c), dpi=150, cmap='gray')
        return si


class Ti(Features):
    def __init__(self):
        super().__init__()
        self.previous_frame = None

    def calc(self, frame):
        self.frame_counter += 1
        ti = 0

        if self.previous_frame is not None:
            difference = frame - self.previous_frame
            ti = round(difference.std(), 4)
            # plt.imshow(np.abs(difference), cmap='gray');plt.show()
            # plt.savefig('temp\\frame' + str(self.frame_counter) + '_diff', dpi=150, cmap='gray')

        # plt.imshow(frame, cmap='gray');plt.show()
        # plt.savefig('temp\\frame' + str(self.frame_counter), dpi=150, cmap='gray')

        self.values.append(ti)
        self.previous_frame = frame
        return ti


if __name__ == "__main__":
    # argument parsing
    parser = argparse.ArgumentParser(description='Calculate SI/TI according ITU-T P.910',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("video", type=str, help="Video to analyze")
    parser.add_argument('--size', type=str, default='0x0', help='Dimensions in pixels')
    parser.add_argument('--num_frames', type=int, default=0, help='Process number of frames')
    parser.add_argument("--output", type=str, default="", help="Output CSV for si ti report")
    parser.add_argument("--format", type=str, default="", help="Force ffmpeg video format")
    parser.add_argument("--pix_fmt", type=str, default="yuv420p", help="force ffmpeg pixel format")
    params = vars(parser.parse_args())

    siti(**params)
