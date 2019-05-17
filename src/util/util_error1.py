from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from util_base import *

import numpy as np
import tensorflow as tf
import tensorflow_compression as tfc

'''
Network Definition of Enhance layer-1
'''
def analysis_transform_e1(tensor, conv_filters_num, num_filters):
  """Builds the analysis transform."""

  with tf.variable_scope("e1_pre256", reuse=tf.AUTO_REUSE):
    with tf.variable_scope("e1_enc_layer_0", reuse=tf.AUTO_REUSE):
      layer = tfc.SignalConv2D(
          conv_filters_num, (9, 9), corr=True, strides_down=4, padding="same_zeros",
          use_bias=True, activation=tfc.GDN())
      tensor = layer(tensor)
      #tf.add_to_collection("e1_pre256", tensor)

    with tf.variable_scope("e1_enc_layer_1", reuse=tf.AUTO_REUSE):
      layer = tfc.SignalConv2D(
          conv_filters_num, (5, 5), corr=True, strides_down=2, padding="same_zeros",
          use_bias=True, activation=tfc.GDN())
      tensor = layer(tensor)
      #tf.add_to_collection("e1_pre256", tensor)

    with tf.variable_scope("e1_enc_layer_2", reuse=tf.AUTO_REUSE):
      layer = tfc.SignalConv2D(
          num_filters, (5, 5), corr=True, strides_down=2, padding="same_zeros",
          use_bias=False, activation=None)
      tensor = layer(tensor)
      #tf.add_to_collection("e1_pre256", tensor)

    return tensor


def synthesis_transform_e1(tensor, conv_filters_num, num_filters):
  """Builds the synthesis transform."""

  with tf.variable_scope("e1_pre256", reuse=tf.AUTO_REUSE):
    with tf.variable_scope("e1_dec_layer_0", reuse=tf.AUTO_REUSE):
      layer = tfc.SignalConv2D(
          conv_filters_num, (5, 5), corr=False, strides_up=2, padding="same_zeros",
          use_bias=True, activation=tfc.GDN(inverse=True))
      tensor = layer(tensor)
      #tf.add_to_collection("e1_pre256", tensor)

    with tf.variable_scope("e1_dec_layer_1", reuse=tf.AUTO_REUSE):
      layer = tfc.SignalConv2D(
          conv_filters_num, (5, 5), corr=False, strides_up=2, padding="same_zeros",
          use_bias=True, activation=tfc.GDN(inverse=True))
      tensor = layer(tensor)
      #tf.add_to_collection("e1_pre256", tensor)

    with tf.variable_scope("e1_dec_layer_2", reuse=tf.AUTO_REUSE):
      layer = tfc.SignalConv2D(
          3, (9, 9), corr=False, strides_up=4, padding="same_zeros",
          use_bias=True, activation=None)
      tensor = layer(tensor)
      #tf.add_to_collection("e1_pre256", tensor)

    return tensor

def entropy_estimation_e1_static(y_e1, istrain):
  with tf.variable_scope("e1_pre256", reuse=tf.AUTO_REUSE):
    entropy_bottleneck_e1 = tfc.EntropyBottleneck()
    y_tilde_e1, likelihoods_e1 = entropy_bottleneck_e1(y_e1, training=istrain)
    string_e1 = entropy_bottleneck_e1.compress(y_e1)

    return y_tilde_e1, likelihoods_e1, entropy_bottleneck_e1

def entropy_estimation_e1(y_e1, istrain):
  with tf.variable_scope("e1_pre256", reuse=tf.AUTO_REUSE):
    entropy_bottleneck_e1 = tfc.EntropyBottleneck()
    y_hat_e1, likelihoods_e1 = entropy_bottleneck_e1(y_e1, training=istrain)
    if istrain:
      return y_hat_e1, likelihoods_e1, entropy_bottleneck_e1
    else:
      string_e1 = entropy_bottleneck_e1.compress(y_e1)
      string_e1 = tf.squeeze(string_e1, axis=0)
      return y_hat_e1, likelihoods_e1, entropy_bottleneck_e1, string_e1


def concate_features_between_layers(tensor, num_filters_en):
  with tf.variable_scope('e1_pre256', reuse=tf.AUTO_REUSE):
    with tf.variable_scope("e1_enc_layer_3", reuse=tf.AUTO_REUSE):
      layer = tfc.SignalConv2D(
          num_filters_en, (1, 1), corr=True, strides_down=1, padding="same_zeros",
          use_bias=False, activation=None)
      tensor = layer(tensor)

  return tensor


'''
Eval code
'''
def eval_test_image_e1(sess, cur_data, test_image, img, y_base, x_hat, x_rec, x_hat_e1, y_hat_e1, eval_bpp, string_e1, q):
    eval_bpp_, string_e1_, x_shape, y_shape = \
        sess.run([eval_bpp, string_e1, tf.shape(img), tf.shape(y_base)], feed_dict={img: cur_data})

    # Write a binary file with the shape information and the compressed string.
    file_name = "./kodak/" + str(q) + "_e1.bin"
    with open(file_name, "wb") as f:
        f.write(np.array(x_shape[1:-1], dtype=np.uint16).tobytes())
        f.write(np.array(y_shape[1:-1], dtype=np.uint16).tobytes())
        f.write(string_e1_)

    # If requested, transform the quantized image back and measure performance.
    x_rec_, x_hat_, x_hat_e1_, y_hat_e1_ = sess.run([x_rec, x_hat, x_hat_e1, y_hat_e1], feed_dict={img: cur_data})

    x_rec_ = np.array(x_rec_)
    x_hat_ = np.array(x_hat_)
    x_hat_e1_plus_base_ = (np.array(x_hat_e1_)  + x_hat_ )
    x_hat_e1_plus_base_ = np.clip(x_hat_e1_plus_base_, 0, 1)
    y_hat_e1_ = np.array(y_hat_e1_)
	
    num_pixels = np.prod(x_hat_e1_plus_base_.shape) * 3
    bpp_base = eval_bpp_
    psnr_r_base = calc_psnr(np.squeeze(test_image[q, :, :, 0]), np.squeeze(x_hat_[:, :, :, 0] ))
    psnr_g_base = calc_psnr(np.squeeze(test_image[q, :, :, 1]), np.squeeze(x_hat_[:, :, :, 1] ))
    psnr_b_base = calc_psnr(np.squeeze(test_image[q, :, :, 2]), np.squeeze(x_hat_[:, :, :, 2] ))
    psnr_base = (psnr_r_base + psnr_g_base + psnr_b_base) / 3

    psnr_r_rec = calc_psnr(np.squeeze(test_image[q, :, :, 0]), np.squeeze(x_rec_[:, :, :, 0] ))
    psnr_g_rec = calc_psnr(np.squeeze(test_image[q, :, :, 1]), np.squeeze(x_rec_[:, :, :, 1] ))
    psnr_b_rec = calc_psnr(np.squeeze(test_image[q, :, :, 2]), np.squeeze(x_rec_[:, :, :, 2] ))
    psnr_rec = (psnr_r_rec + psnr_g_rec + psnr_b_rec) / 3

    bpp_e1 = (8 + len(string_e1_)) * 8 / num_pixels
    psnr_r_e1 = calc_psnr_e1(np.squeeze(test_image[q, :, :, 0]), np.squeeze(x_hat_e1_plus_base_[:, :, :, 0]))
    psnr_g_e1 = calc_psnr_e1(np.squeeze(test_image[q, :, :, 1]), np.squeeze(x_hat_e1_plus_base_[:, :, :, 1]))
    psnr_b_e1 = calc_psnr_e1(np.squeeze(test_image[q, :, :, 2]), np.squeeze(x_hat_e1_plus_base_[:, :, :, 2]))
    psnr_e1_plus_base = (psnr_r_e1 + psnr_g_e1 + psnr_b_e1) / 3

    msssim_r_rec = calc_ms_ssim_e1(np.squeeze(test_image[q, :, :, 0]), np.squeeze(x_rec_[:, :, :, 0]))
    msssim_g_rec = calc_ms_ssim_e1(np.squeeze(test_image[q, :, :, 1]), np.squeeze(x_rec_[:, :, :, 1]))
    msssim_b_rec = calc_ms_ssim_e1(np.squeeze(test_image[q, :, :, 2]), np.squeeze(x_rec_[:, :, :, 2]))
    msssim_rec = (msssim_r_rec + msssim_g_rec + msssim_b_rec) / 3
    if os.path.exists(file_name):
        os.remove(file_name)

    return psnr_rec, psnr_base, psnr_e1_plus_base, bpp_base, bpp_e1, x_hat_, x_hat_e1_plus_base_, y_hat_e1_,msssim_rec

def calc_psnr_e1(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    return 10 * np.log10(1 / mse)

def calc_ms_ssim_e1(img1, img2):
    image1 = tf.placeholder(tf.float32, shape=[1, img1.shape[0], img1.shape[1], 1])
    image2 = tf.placeholder(tf.float32, shape=[1, img2.shape[0], img2.shape[1], 1])
    msssim = tf.image.ssim_multiscale(image1, image2, max_val=1.0)
    img1 = np.reshape(img1, [1, img1.shape[0], img1.shape[1], 1])
    img2 = np.reshape(img2, [1, img2.shape[0], img2.shape[1], 1])
    with tf.Session() as sess:
        tf_msssim = sess.run([msssim], feed_dict={image1: img1, image2: img2})
    tf_msssim = np.float32(tf_msssim)
    return tf_msssim
