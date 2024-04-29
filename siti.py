#!/usr/bin/env python3
import argparse
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage


class SiTi:
    """
    Calculate the Spatial information and Temporal information according
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

    def __init__(self, video: str, output: str, n_frames: int = None, verbose=False):
        self.video = Path(video)
        self.output = Path(output)
        self.n_frames = n_frames
        self.verbose = verbose

        self.si = []
        self.ti = []
        self.previous_frame = None
        self.frame_counter = 0
        self.stats = defaultdict(list)

    def calc_siti(self):
        """
        Start the calculation of the SI and TI. The values are stored in the 'si'
        and 'ti' attributes for each frame.

        :return: None
        """

        for self.frame_counter, frame in enumerate(self._iter_video(), 1):
            if self.frame_counter > self.n_frames:
                break

            si = self._calc_si(frame)
            self.si.append(si)

            ti = self._calc_ti(frame)
            if ti != float('inf'):
                self.ti.append(ti)

            if self.verbose:
                print(f"{self.frame_counter:04}, "
                      f"si={si:05.3f}, ti={ti:05.3f}")
            else:
                if self.frame_counter % 10 == 0:
                    print('.', end='', flush=True)

        self.create_stats()
        return self.si, self.ti

    def _iter_video(self):
        cap = cv2.VideoCapture(f'{self.video}')
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            yield gray.astype(float)

    @staticmethod
    def _sobel(frame):
        """
        Apply 1st order 2D Sobel filter
        :param frame:
        :return:
        """
        sob_x = ndimage.sobel(frame, axis=0)
        sob_y = ndimage.sobel(frame, axis=1)
        sob = np.hypot(sob_x, sob_y)
        return sob

    def _calc_si(self, frame: np.ndarray) -> (float, np.ndarray):
        """
        Calculate Spatial Information for a video frame.
        :param frame: A luma video frame in numpy ndarray format.
        :return: spatial information and sobel frame.
        """
        sobel = self._sobel(frame)
        si = sobel.std()
        return si

    def _calc_ti(self, frame: np.ndarray) -> (float, np.ndarray):
        """
        Calculate Temporal Information for a video frame. If is a first frame,
        the information is zero.
        :param frame: A luma video frame in numpy ndarray format.
        :return: Temporal information and difference frame. If first frame the
        difference is zero array on same shape of frame.
        """
        if self.previous_frame is not None:
            difference = frame - self.previous_frame
            ti = difference.std()
        else:
            ti = float('inf')

        self.previous_frame = frame

        return ti

    def save_siti(self):
        filename = self.video.with_suffix('.csv')
        df = pd.DataFrame({'si': self.si, 'ti': [0]+self.ti},
                          index=range(len(self.si)))
        df.to_csv(f'{filename}', index_label='frame')

    def create_stats(self):
        self.stats['si_average'].append(f'{np.average(self.si):05.3f}')
        self.stats['si_std'].append(f'{np.std(self.si):05.3f}')
        self.stats['si_0q'].append(f'{np.quantile(self.si, 0.00):05.3f}')
        self.stats['si_1q'].append(f'{np.quantile(self.si, 0.25):05.3f}')
        self.stats['si_2q'].append(f'{np.quantile(self.si, 0.50):05.3f}')
        self.stats['si_3q'].append(f'{np.quantile(self.si, 0.75):05.3f}')
        self.stats['si_4q'].append(f'{np.quantile(self.si, 1.00):05.3f}')

        self.stats['ti_average'].append(f'{np.average(self.ti):05.3f}')
        self.stats['ti_std'].append(f'{np.std(self.ti):05.3f}')
        self.stats['ti_0q'].append(f'{np.quantile(self.ti, 0.00):05.3f}')
        self.stats['ti_1q'].append(f'{np.quantile(self.ti, 0.25):05.3f}')
        self.stats['ti_2q'].append(f'{np.quantile(self.ti, 0.50):05.3f}')
        self.stats['ti_3q'].append(f'{np.quantile(self.ti, 0.75):05.3f}')
        self.stats['ti_4q'].append(f'{np.quantile(self.ti, 1.00):05.3f}')

    def save_stats(self):
        pd.DataFrame(self.stats).to_csv(f'{self.output}', index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='SiTi',
        description='Calculate SI/TI for all frames according ITU-T P.910, and creates average, '
                    'standard deviation, maximum, minimum and quartile statistics.'
                    ' '
                    'The file must be in a container format, such as MP4, AVI, MKV, etc., '
                    'for OpenCV to read the metadata. Convert the file using the x264 '
                    'encoder without FFmpeg with the CRF parameter set to 0. For example:'
                    ' '
                    'ffmpeg -video_size 960x480 -framerate 25 -pixel_format yuv420p -format '
                    'rawvideo -i input_video.yuv -crf 0 output_video.mp4'
                    ' '
                    'See: https://trac.ffmpeg.org/wiki/Encode/H.264, and https://ffmpeg.org/ffmpeg.html',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-o', '--output',
                        type=str,
                        default='output.csv',
                        help='Output CSV for si ti report and statistics. Default: output.csv.')
    parser.add_argument('-n', '--n_frames',
                        type=int,
                        default=None,
                        help='Process n frames.')
    parser.add_argument('-v', '--verbose',
                        action='store_true',
                        help="Show message in terminal.")
    parser.add_argument('video',
                        type=str,
                        help="Video input to analyze.")

    params = vars(parser.parse_args())

    siti = SiTi(**params)
    siti.calc_siti()
    siti.save_siti()
    siti.save_stats()
    print('###########################')
    print('######### RESULTS #########')
    print('###########################')
    for stat, (value,) in siti.stats.items():
        print(f'\t{stat}:\t{value}')
