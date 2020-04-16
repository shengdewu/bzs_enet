'''
Semantic Instance Segmentation with a Discriminative Loss Function
'''
import tensorflow as tf

def discriminative_loss(label, predict, label_shape, feature_dim, delta_v, delta_d, param_var, param_dist, param_reg):
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
    distance = tf.subtract(distance, delta_v)
    distance = tf.clip_by_value(distance, 0, distance)
    distance = tf.square(distance)

    l_var = tf.unsorted_segment_sum(distance, label_index, instance_num)
    l_var = tf.reduce_sum(tf.div(l_var, label_cnt))
    l_var = tf.div(l_var, tf.cast(instance_num,tf.float32))

    uc_avg_1 = tf.tile(uc_avg, [instance_num, 1])
    shape = uc_avg_1.get_shape().as_list()
    uc_avg_2 = tf.tile(uc_avg, [1, instance_num])
    uc_avg_2 = tf.reshape(uc_avg_2, (instance_num*instance_num, feature_dim))
    uc_diff = tf.subtract(uc_avg_1, uc_avg_2)

    uc_sum_cache = tf.reduce_sum(tf.abs(uc_diff), axis=1)
    zero_vector = tf.zeros(1, dtype=tf.float32)
    bool_mask = tf.not_equal(uc_sum_cache, zero_vector)
    mu_diff_bool = tf.boolean_mask(uc_diff, bool_mask)

    mu_norm1 = tf.norm(mu_diff_bool, axis=1)
    mu_norm1 = tf.subtract(2.*delta_d, mu_norm1)
    mu_norm_clip = tf.clip_by_value(mu_norm1, 0., mu_norm1)
    mu_norm = tf.square(mu_norm_clip)

    l_dist = tf.reduce_mean(mu_norm)

    ### Calculate l_reg
    l_reg = tf.reduce_mean(tf.norm(uc_avg, axis=1))

    param_scale = 1.
    l_var = param_var * l_var
    l_dist = param_dist * l_dist
    l_reg = param_reg * l_reg

    loss = param_scale*(l_var + l_dist + l_reg)

    return loss


def discriminative_loss_batch(prediction, correct_label, feature_dim, image_shape,
                        delta_v, delta_d, param_var, param_dist, param_reg):

    def cond(label, batch, out_loss, out_var, out_dist, out_reg, i):
        return tf.less(i, tf.shape(batch)[0])

    def body(label, batch, out_loss, out_var, out_dist, out_reg, i):
        disc_loss, l_var, l_dist, l_reg = discriminative_loss(
            correct_label[i], prediction[i], image_shape, feature_dim, delta_v, delta_d, param_var, param_dist, param_reg)

        out_loss = out_loss.write(i, disc_loss)
        out_var = out_var.write(i, l_var)
        out_dist = out_dist.write(i, l_dist)
        out_reg = out_reg.write(i, l_reg)

        return label, batch, out_loss, out_var, out_dist, out_reg, i + 1

    # TensorArray is a data structure that support dynamic writing
    output_ta_loss = tf.TensorArray(
        dtype=tf.float32, size=0, dynamic_size=True)
    output_ta_var = tf.TensorArray(
        dtype=tf.float32, size=0, dynamic_size=True)
    output_ta_dist = tf.TensorArray(
        dtype=tf.float32, size=0, dynamic_size=True)
    output_ta_reg = tf.TensorArray(
        dtype=tf.float32, size=0, dynamic_size=True)

    _, _, out_loss_op, out_var_op, out_dist_op, out_reg_op, _ = tf.while_loop(
        cond, body, [
            correct_label, prediction, output_ta_loss, output_ta_var, output_ta_dist, output_ta_reg, 0])
    out_loss_op = out_loss_op.stack()
    out_var_op = out_var_op.stack()
    out_dist_op = out_dist_op.stack()
    out_reg_op = out_reg_op.stack()

    disc_loss = tf.reduce_mean(out_loss_op)
    l_var = tf.reduce_mean(out_var_op)
    l_dist = tf.reduce_mean(out_dist_op)
    l_reg = tf.reduce_mean(out_reg_op)

    return disc_loss, l_var, l_dist, l_reg