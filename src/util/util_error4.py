from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from util_base import *

import numpy as np
import tensorflow as tf
import tensorflow_compression as tfc

'''
Network Definition of Enhance layer-4
'''
def analysis_transform_e4(tensor, conv_filters_num, num_filters):
  """Builds the analysis transform."""

  with tf.variable_scope("e4_pre256", reuse=tf.AUTO_REUSE):
    with tf.variable_scope("e4_enc_layer_0", reuse=tf.AUTO_REUSE):
      layer = tfc.SignalConv2D(
          conv_filters_num, (9, 9), corr=True, strides_down=4, padding="same_zeros",
          use_bias=True, activation=tfc.GDN())
      tensor = layer(tensor)

    with tf.variable_scope("e4_enc_layer_1", reuse=tf.AUTO_REUSE):
      layer = tfc.SignalConv2D(
          conv_filters_num, (5, 5), corr=True, strides_down=2, padding="same_zeros",
          use_bias=True, activation=tfc.GDN())
      tensor = layer(tensor)

    with tf.variable_scope("e4_enc_layer_2", reuse=tf.AUTO_REUSE):
      layer = tfc.SignalConv2D(
          num_filters, (5, 5), corr=True, strides_down=2, padding="same_zeros",
          use_bias=False, activation=None)
      tensor = layer(tensor)

    return tensor


def synthesis_transform_e4(tensor, conv_filters_num, num_filters):
  """Builds the synthesis transform."""

  with tf.variable_scope("e4_pre256", reuse=tf.AUTO_REUSE):
    with tf.variable_scope("e4_dec_layer_0", reuse=tf.AUTO_REUSE):
      layer = tfc.SignalConv2D(
          conv_filters_num, (5, 5), corr=False, strides_up=2, padding="same_zeros",
          use_bias=True, activation=tfc.GDN(inverse=True))
      tensor = layer(tensor)

    with tf.variable_scope("e4_dec_layer_1", reuse=tf.AUTO_REUSE):
      layer = tfc.SignalConv2D(
          conv_filters_num, (5, 5), corr=False, strides_up=2, padding="same_zeros",
          use_bias=True, activation=tfc.GDN(inverse=True))
      tensor = layer(tensor)

    with tf.variable_scope("e4_dec_layer_2", reuse=tf.AUTO_REUSE):
      layer = tfc.SignalConv2D(
          3, (9, 9), corr=False, strides_up=4, padding="same_zeros",
          use_bias=True, activation=None)
      tensor = layer(tensor)

    return tensor

def entropy_estimation_e4(y_e4, istrain):
  with tf.variable_scope("e4_pre256", reuse=tf.AUTO_REUSE):
    entropy_bottleneck_e4 = tfc.EntropyBottleneck()
    y_hat_e4, likelihoods_e4 = entropy_bottleneck_e4(y_e4, training=istrain)
    if istrain:
      return y_hat_e4, likelihoods_e4, entropy_bottleneck_e4
    else:
      string_e4 = entropy_bottleneck_e4.compress(y_e4)
      string_e4 = tf.squeeze(string_e4, axis=0)
      return y_hat_e4, likelihoods_e4, entropy_bottleneck_e4, string_e4


def concate_features_between_layers_e4(tensor, num_filters):
  with tf.variable_scope('e4_pre256', reuse=tf.AUTO_REUSE):
    with tf.variable_scope("e4_enc_layer_3", reuse=tf.AUTO_REUSE):
      layer = tfc.SignalConv2D(
          num_filters, (1, 1), corr=True, strides_down=1, padding="same_zeros",
          use_bias=False, activation=None)
      tensor = layer(tensor)

  return tensor


'''
Eval code
'''
def eval_test_image_e4(sess, x_rec, cur_data, test_image, img, y_e4, x_hat_e4, x_hat_e3_e2_e1_base, string_e4, q):
    string_e4_, x_shape, y_shape = \
        sess.run([string_e4, tf.shape(img), tf.shape(y_e4)], feed_dict={img: cur_data})

    # Write a binary file with the shape information and the compressed string.
    file_name = "./kodak/" + str(q) + "_e4.bin"
    with open(file_name, "wb") as f:
        f.write(np.array(x_shape[1:-1], dtype=np.uint16).tobytes())
        f.write(np.array(y_shape[1:-1], dtype=np.uint16).tobytes())
        f.write(string_e4_)

    # If requested, transform the quantized image back and measure performance.
    x_hat_e4_ = sess.run([x_hat_e4], feed_dict={img: cur_data})
    x_hat_e4_ = np.reshape(x_hat_e4_, [1, wid, hgt, channels])
    x_hat_e3_e2_e1_base = np.reshape(x_hat_e3_e2_e1_base, [1, wid, hgt, channels])

    x_hat_e4_e3_e2_e1_base_ = (np.array(x_hat_e4_)  + x_hat_e3_e2_e1_base)
    x_hat_e4_e3_e2_e1_base_ = np.clip(x_hat_e4_e3_e2_e1_base_, 0, 1)

    x_rec_ = sess.run([x_rec], feed_dict={img: cur_data})
    x_rec_ = np.reshape(x_rec_, [1, wid, hgt, channels])
    x_rec_ = np.clip(x_rec_, 0, 1)

    num_pixels = np.prod(x_hat_e4_e3_e2_e1_base_.shape) * 3

    bpp_e4 = (8 + len(string_e4_)) * 8 / num_pixels
    psnr_r_e4 = calc_psnr_e4(np.squeeze(test_image[q, :, :, 0]), np.squeeze(x_hat_e4_e3_e2_e1_base_[:, :, :, 0]))
    psnr_g_e4 = calc_psnr_e4(np.squeeze(test_image[q, :, :, 1]), np.squeeze(x_hat_e4_e3_e2_e1_base_[:, :, :, 1]))
    psnr_b_e4 = calc_psnr_e4(np.squeeze(test_image[q, :, :, 2]), np.squeeze(x_hat_e4_e3_e2_e1_base_[:, :, :, 2]))
    psnr_e4_e3_e2_e1_base = (psnr_r_e4 + psnr_g_e4 + psnr_b_e4) / 3

    psnr_r_rec = calc_psnr_e4(np.squeeze(test_image[q, :, :, 0]), np.squeeze(x_rec_[:, :, :, 0]))
    psnr_g_rec = calc_psnr_e4(np.squeeze(test_image[q, :, :, 1]), np.squeeze(x_rec_[:, :, :, 1]))
    psnr_b_rec = calc_psnr_e4(np.squeeze(test_image[q, :, :, 2]), np.squeeze(x_rec_[:, :, :, 2]))
    psnr_rec = (psnr_r_rec + psnr_g_rec + psnr_b_rec) / 3

    msssim_r_rec = calc_ms_ssim_e4(np.squeeze(test_image[q, :, :, 0]), np.squeeze(x_rec_[:, :, :, 0]))
    msssim_g_rec = calc_ms_ssim_e4(np.squeeze(test_image[q, :, :, 1]), np.squeeze(x_rec_[:, :, :, 1]))
    msssim_b_rec = calc_ms_ssim_e4(np.squeeze(test_image[q, :, :, 2]), np.squeeze(x_rec_[:, :, :, 2]))
    msssim_rec = (msssim_r_rec + msssim_g_rec + msssim_b_rec) / 3

    if os.path.exists(file_name):
        os.remove(file_name)

    return psnr_rec, psnr_e4_e3_e2_e1_base, bpp_e4, x_hat_e4_e3_e2_e1_base_, msssim_rec

def calc_psnr_e4(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    return 10 * np.log10(1 / mse)

def calc_ms_ssim_e4(img1, img2):
    image1 = tf.placeholder(tf.float32, shape=[1, img1.shape[0], img1.shape[1], 1])
    image2 = tf.placeholder(tf.float32, shape=[1, img2.shape[0], img2.shape[1], 1])
    msssim = tf.image.ssim_multiscale(image1, image2, max_val=1.0)
    img1 = np.reshape(img1, [1, img1.shape[0], img1.shape[1], 1])
    img2 = np.reshape(img2, [1, img2.shape[0], img2.shape[1], 1])
    with tf.Session() as sess:
        tf_msssim = sess.run([msssim], feed_dict={image1: img1, image2: img2})
    tf_msssim = np.float32(tf_msssim)
    return tf_msssim

