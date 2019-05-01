import glob
import siti


def main():
    for filename in glob.glob('*.yuv'):
        siti.siti(filename=filename, size='4320x2160', form='rawvideo')

    siti.multi_plot('*.csv')


if __name__ == "__main__":
    main()
