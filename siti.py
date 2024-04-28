#!/usr/bin/env python3
import argparse

import numpy as np
import pandas as pd
import skvideo.io
from scipy import ndimage
import os


class SiTi:
    """
    Calcule the Spatial information and Temporal information according
    ITU-T P.910, but using median instead of the maximum value. In the
    recommendation, the SI/TI for the video is the maximum value. But, for long
    videos the values vary a lot. In this way, the median will represent that
    half of frame values are below and half will be above.

    The Spatial Information.
    The spatial information is based on the Sobel Filter on the luminance
    frame. For each frame (time dimension) filtered with the Sobel filter, the
    standard deviation is computed (space dimension). The Spatial Information
    for the video is the median of the all frames. On the ITU-T P.910
    recommendation the max value is used. So, large variations in luminance will
    generate higher values for SI.

    SI = median_time(std_space(sobel(Frame_n)))

    The Temporal Information
    The temporal information is based on the motion difference. For each
    luminance frame pair (n, n+1) the difference is computed and the Standard
    Deviation is calculated. The Temporal Information is the Median of the all
    frames. On the ITU-T P.910 recommendation the max value is used. So, large
    variations in movement (large differences between frames) will produce
    larger TI.

    TI = median_time(std_space(difference(Frame_n, Frame_n-1)))


    """

    def __init__(self, filename, scale, format=None, pix_fmt=None):
        """

        :param filename: The filename of video in path-like format.
        :param scale: The frame dimension in string format. e.g.: "640x480".
        :param format: force ffmpeg format. E.g.: "yuv", "mp4", "hevc", etc. See
        "ffmpeg -formats".
        :param pix_fmt: force ffmpeg pixel format. E.g.: "yuv420p", "yuv444p",
        "rgb8", etc.
        """
        self.filename = filename
        self.scale = scale
        self.width, self.height = tuple(map(int, scale.split('x')))

        self.si = []
        self.ti = []
        self.previous_frame = None
        self.frame_counter = 0

        self.inputdict = {'-s': f'{self.scale}'}

        if format:
            self.inputdict["-f"] = form
        if pix_fmt:
            self.inputdict["-pix_fmt"] = pix_fmt

    @staticmethod
    def sobel(frame):
        """
        Apply 1st order 2D Sobel filter
        :param frame:
        :return:
        """
        sobx = ndimage.sobel(frame, axis=0)
        soby = ndimage.sobel(frame, axis=1)
        sob = np.hypot(sobx, soby)
        return sob

    def _calc_si(self, frame: np.ndarray) -> (float, np.ndarray):
        """
        Calcule Spatial Information for a video frame.
        :param frame: A luma video frame in numpy ndarray format.
        :return: spatial information and sobel frame.
        """
        sobel = self.sobel(frame)
        si = sobel.std()
        self.si.append(si)
        return si, sobel

    def _calc_ti(self, frame: np.ndarray) -> (float, np.ndarray):
        """
        Calcule Temporal Information for a video frame. If is a first frame,
        the information is zero.
        :param frame: A luma video frame in numpy ndarray format.
        :return: Temporal information and diference frame. If first frame the
        diference is zero array on same shape of frame.
        """
        if self.previous_frame:
            difference = frame - self.previous_frame
            ti = difference.std()
        else:
            difference = np.zeros(frame.shape)
            ti = 0.0

        self.ti.append(ti)
        self.previous_frame = frame

        return ti, difference

    def calc_siti(self, verbose=False, n_frames=None):
        """
        Start the calculation of the SI and TI. The values is stored in the 'si'
        and 'ti' attributes for each frame.

        :param verbose: Show calculation frame-by-frame.
        :param n_frames: Frames to be processed.

        :return: None
        """

        vreader = skvideo.io.vreader(fname=self.filename, as_grey=True,
                                     inputdict=self.inputdict)

        for self.frame_counter, frame in enumerate(vreader, 1):
            if self.frame_counter > n_frames:
                break
            width = frame.shape[1]
            height = frame.shape[2]
            frame = frame.reshape((width, height)).astype('float32')
            value_si, sobel = self._calc_si(frame)
            value_ti, difference = self._calc_ti(frame)
            if verbose:
                print(f"{self.frame_counter:04}, "
                      f"si={value_si:05.3f}, ti={value_ti:05.3f}")
            else:
                print('.', end='', flush=True)

    def save_siti(self, filename=None):
        if not filename:
            filename, ext = os.path.splitext(self.filename)
            filename = filename + '.csv'

        df = pd.DataFrame({'si': self.si, 'ti': self.ti},
                          index=range(len(self.si)))
        df.to_csv(f'{filename}.csv', index_label='frame')

    def save_stats(self, filename=None):
        if not filename:
            filename, ext = os.path.splitext(self.filename)
            filename = filename + '.csv'

        stats = dict(si_average=f'{np.average(self.si):05.3f}',
                     si_std=f'{np.std(self.si):05.3f}',
                     si_0q=f'{np.quantile(self.si, 0.00):05.3f}',
                     si_1q=f'{np.quantile(self.si, 0.25):05.3f}',
                     si_2q=f'{np.quantile(self.si, 0.50):05.3f}',
                     si_3q=f'{np.quantile(self.si, 0.75):05.3f}',
                     si_4q=f'{np.quantile(self.si, 1.00):05.3f}',
                     ti_average=f'{np.average(self.ti):05.3f}',
                     ti_std=f'{np.std(self.ti):05.3f}',
                     ti_0q=f'{np.quantile(self.ti, 0.00):05.3f}',
                     ti_1q=f'{np.quantile(self.ti, 0.25):05.3f}',
                     ti_2q=f'{np.quantile(self.ti, 0.50):05.3f}',
                     ti_3q=f'{np.quantile(self.ti, 0.75):05.3f}',
                     ti_4q=f'{np.quantile(self.ti, 1.00):05.3f}')

        name = os.path.basename(filename)
        df = pd.DataFrame({name: stats}, index=range(len(self.si)))
        df.to_csv(f'{filename}', index_label='Stats')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Calculate SI/TI according ITU-T P.910, but using median instead of the maximum value.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('video', type=str, help="Video to analyze")
    parser.add_argument('--num_frames', type=int, default=0,
                        help='Process number of frames')
    parser.add_argument('--output', type=str, default="",
                        help="Output CSV for si ti report and statistics")
    parser.add_argument('--size', type=str, default='0x0',
                        help='Dimensions in pixels')
    parser.add_argument('--form', type=str, default="",
                        help="Force ffmpeg video format")
    parser.add_argument("--pix_fmt", type=str, default="yuv420p",
                        help="force ffmpeg pixel format")
    params = vars(parser.parse_args())

    video = params['video']
    size = params['size']
    num_frames = params['num_frames']
    output = params['output']
    form = params['form']
    pix_fmt = params['pix_fmt']

    siti = SiTi(video, size)
    siti.calc_siti()
