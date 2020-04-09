'''
Semantic Instance Segmentation with a Discriminative Loss Function
'''
import tensorflow as tf

def discriminative_loss(label, predict, label_shape, feature_dim, param_var, param_dist, param_reg, param_v, param_d):
    label_reshape = tf.reshape(label, shape=[label_shape[0]*label_shape[1]])
    predict_reshape = tf.reshape(predict, shape=[label_shape[0]*label_shape[1], feature_dim])
    predict_reshape = tf.cast(predict_reshape, tf.float32)

    #统计 类别的标签(unique_label) 类别标签在label对应的索引(label_index) 类别标签对应的个数(label_cnt)
    unique_label, label_index, label_cnt = tf.unique_with_counts(label_reshape)
    label_cnt = tf.cast(label_cnt, tf.float32)
    instance_num = tf.size(unique_label)

    #计算 uc 即每个类别在标签中的均值 即聚类中心
    uc = tf.unsorted_segment_sum(predict_reshape, label_index, instance_num)
    uc_avg = tf.div(uc, tf.reshape(label_cnt, (-1, 1)))
    uc_avg_expand = tf.gather(uc_avg, label_index)

    distance = tf.norm(tf.subtract(uc_avg_expand, predict_reshape), axis=1)
    distance = tf.subtract(distance, param_v)
    distance = tf.clip_by_value(distance, 0, distance)
    distance = tf.square(distance)

    l_var = tf.unsorted_segment_sum(distance, label_index, instance_num)
    l_var = tf.reduce_sum(tf.div(l_var, label_cnt))
    l_var = tf.div(l_var, tf.cast(instance_num,tf.float32))

    return label_reshape, predict_reshape, unique_label, label_index, label_cnt, uc, uc_avg, uc_avg_expand, distance, l_var