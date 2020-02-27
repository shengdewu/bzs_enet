from lannet.lannet import lannet

if __name__ == '__main__':
    lannet_model = lannet()
    lannet_model.train()

    print('train success!!')