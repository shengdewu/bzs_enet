from tusimple_process.lanenet_data_pipline import lanenet_data_pipline
import sys

if __name__ == '__main__':
    lanenet_data_provide = lanenet_data_pipline()
    tusimple_path = sys.argv[1]
    if sys.argv[3] == 'test':
        lanenet_data_provide.generate_test_data(tusimple_path)
    else:
        out_path = sys.argv[2]
        lanenet_data_provide.generate_data(tusimple_path, out_path)