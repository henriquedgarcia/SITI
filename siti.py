#!/usr/bin/env python3

import argparse
import platform
import numpy as np
import pandas as pd
import skvideo.io
from scipy import ndimage
import matplotlib.pyplot as plt
import glob

if platform.system() == 'Windows':
    sl = '\\'
else:
    sl = '/'


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
    print(f'Calculating SI/TI for {filename}.')
    frame_counter = 0
    [width, height] = list(map(int, size.split('x')))
    if output in '':
        output = '.'.join(filename.split('.')[:-1]) + '.csv'  # Remove extension
    if form not in '' and inputdict is None:
        inputdict = {"-s": f"{width}x{height}", "-pix_fmt": pix_fmt, '-f': form}

    measures = {'SI': Si(), 'TI': Ti()}

    video = skvideo.io.vreader(fname=filename, inputdict=inputdict, as_grey=True, num_frames=num_frames)

    for frame in video:
        frame_counter += 1
        print(f"\nframe {frame_counter} of video {video}")
        frame = frame.reshape((height, width)).astype('float32')

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
        si = sob_std
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
            ti = difference.std()
            # plt.imshow(np.abs(difference), cmap='gray');plt.show()
            # plt.savefig('temp\\frame' + str(self.frame_counter) + '_diff', dpi=150, cmap='gray')

        # plt.imshow(frame, cmap='gray');plt.show()
        # plt.savefig('temp\\frame' + str(self.frame_counter), dpi=150, cmap='gray')

        self.values.append(ti)
        self.previous_frame = frame
        return ti


def multi_plot(input_glob='*.csv'):
    names = {}

    # For each file plot
    for x in glob.glob(input_glob):
        name = x.replace(f'.{input_glob.split(".")[-1]}', '')
        tmp = pd.read_csv(x, delimiter=',')
        names[name] = [tmp['si'].max(),
                       tmp['ti'].max(),
                       tmp['si'].mean(),
                       tmp['ti'].mean(),
                       tmp['si'].median(),
                       tmp['ti'].median()]

        fig, ax = plt.subplots(figsize=(9, 5), tight_layout=True, dpi=300)
        ax.plot(tmp['si'], label='si')
        ax.plot(tmp['ti'], label='ti')
        ax.set_xlabel('Frame')
        ax.set_ylabel('Information')
        ax.set_title(name)
        ax.legend(loc='upper left', bbox_to_anchor=(1.01, 0.99))
        plt.show()
        fig.savefig(name)

    # A geral plot
    fig, [ax_max, ax_avg, ax_med] = plt.subplots(1, 3, figsize=(18, 5), tight_layout=True, dpi=200)
    for name in names:
        ax_max.scatter(names[name][0], names[name][1], label=name)
        ax_avg.scatter(names[name][2], names[name][3], label=name)
        ax_med.scatter(names[name][4], names[name][5], label=name)

    ax_max.set_xlabel('SI')
    ax_max.set_ylabel('TI')
    ax_max.set_title('SI/TI - Max Values')
    ax_avg.set_xlabel('SI')
    ax_avg.set_ylabel('TI')
    ax_avg.set_title('SI/TI - Average Values')
    ax_med.set_xlabel('SI')
    ax_med.set_ylabel('TI')
    ax_med.set_title('SI/TI - Median Values')
    ax_med.legend(loc='upper left', bbox_to_anchor=(1.01, 0.99))
    # plt.show()
    fig.savefig('scatter_siti')


if __name__ == "__main__":
    # argument parsing
    parser = argparse.ArgumentParser(description='Calculate SI/TI according ITU-T P.910',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("filename", type=str, help="Video to analyze")
    parser.add_argument('--size', type=str, default='0x0', help='Dimensions in pixels')
    parser.add_argument('--num_frames', type=int, default=0, help='Process number of frames')
    parser.add_argument("--output", type=str, default="", help="Output CSV for si ti report")
    parser.add_argument("--form", type=str, default="", help="Force ffmpeg video format")
    parser.add_argument("--pix_fmt", type=str, default="yuv420p", help="force ffmpeg pixel format")
    params = vars(parser.parse_args())

    siti(**params)
    multi_plot(params['output'])
