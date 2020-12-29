import tensorflow as tf

def similaryit_loss(label):
    batch, rows, lanes, cells = label.get_shape()
    loss_all = list()
    for i in range(rows-1):
        loss_all.append(tf.abs(label[:,i,:,0:cells-1] - label[:,i+1,:,0:cells-1]))

    loss = tf.concat(loss_all, 0)
    less_one = tf.cast(tf.less(loss, 1.0), tf.float32)
    smooth_l1_loss = (less_one * 0.5 * loss **2) + (1.0-less_one) * (loss-0.5)
    return tf.reduce_mean(smooth_l1_loss)

def structural_loss(label):
    batch, rows, lanes, cells = label.get_shape()
    prob = tf.nn.softmax(label[:,:,:,0:cells-1], -1)
    k = tf.convert_to_tensor([i for i in range(1, cells)], dtype=tf.float32)
    loc = tf.reduce_sum(prob * k, -1)
    loss_all = list()
    for i in range(rows//2):
        loss_all.append(tf.abs((loc[:,i,:]-loc[:,i+1,:])-(loc[:,i+1,:]-loc[:,i+2,:])))
    loss = tf.concat(loss_all, 0)
    return tf.reduce_mean(loss)

def cls_loss(group_cls, label, cells):
    bs, ws, hs, cs = label.get_shape().as_list()
    #多分类交叉熵
    #-Ybc * log(Pbc)
    #sum(c)
    #sum(b)/b
    scores = tf.nn.softmax(group_cls, axis=3)
    factor = tf.pow(1.-scores, 2)
    log_score = tf.nn.log_softmax(group_cls, axis=3)
    log_score = factor * log_score

    label = tf.reshape(label, (bs, ws, hs))
    label_oh = tf.one_hot(label, cells+1)
    nllloss1 = tf.multiply(label_oh, log_score)
    nllloss2 = tf.abs(nllloss1)
    index = tf.where(nllloss2 > 0)
    nllloss3 = tf.gather_nd(nllloss2, index)

    cls = tf.reduce_mean(nllloss3)

    return cls
