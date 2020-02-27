from lannet.lannet import lannet
import sys

if __name__ == '__main__':
    lannet_model = lannet()
    lannet_model.train(sys.argv[1])

    print('train success!!')