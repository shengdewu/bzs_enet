from ultra_lane.ultra_lane import ultra_lane
import tensorflow as tf

if __name__ == '__main__':
    lannet_model = ultra_lane()

    x = tf.get_variable(name='x', shape=(10, 800, 288, 3))
    net = lannet_model.make_net(x, 100, 56, 4)
    print('infer success!!')