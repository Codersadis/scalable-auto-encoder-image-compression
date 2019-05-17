from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import glob as gb
import numpy as np
import tensorflow as tf
import matplotlib.image as mpimg
import tensorflow_compression as tfc
channels = 3
wid = 512
hgt = 768
size = 256

def load_image(filename):
  """Loads a PNG image file."""

  string = tf.read_file(filename)
  image = tf.image.decode_image(string, channels=3)
  image = tf.cast(image, tf.float32)
  image /= 255
  return image


def save_image(filename, image):
  """Saves an image to a PNG file."""

  image = tf.clip_by_value(image, 0, 1)
  image = tf.round(image * 255)
  image = tf.cast(image, tf.uint8)
  string = tf.image.encode_png(image)
  return tf.write_file(filename, string)


def calc_psnr(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    return 10 * np.log10(1 / mse)

def calc_ms_ssim_base(img1, img2):
    image1 = tf.placeholder(tf.float32, shape=[1, img1.shape[0], img1.shape[1], 1])
    image2 = tf.placeholder(tf.float32, shape=[1, img2.shape[0], img2.shape[1], 1])
    msssim = tf.image.ssim_multiscale(image1, image2, max_val=1.0)
    img1 = np.reshape(img1, [1, img1.shape[0], img1.shape[1], 1])
    img2 = np.reshape(img2, [1, img2.shape[0], img2.shape[1], 1])
    with tf.Session() as sess:
        tf_msssim = sess.run([msssim], feed_dict={image1: img1, image2: img2})
    tf_msssim = np.float32(tf_msssim)
    return tf_msssim

def analysis_transform(tensor, conv_filters_num, num_filters):
  """Builds the analysis transform."""

  with tf.variable_scope("pre256", reuse=tf.AUTO_REUSE):
    with tf.variable_scope("enc_layer_0", reuse=tf.AUTO_REUSE):
      layer = tfc.SignalConv2D(
          conv_filters_num, (9, 9), corr=True, strides_down=4, padding="same_zeros",
          use_bias=True, activation=tfc.GDN())
      tensor = layer(tensor)
      tf.add_to_collection("pre256", tensor)

    with tf.variable_scope("enc_layer_1", reuse=tf.AUTO_REUSE):
      layer = tfc.SignalConv2D(
          conv_filters_num, (5, 5), corr=True, strides_down=2, padding="same_zeros",
          use_bias=True, activation=tfc.GDN())
      tensor = layer(tensor)
      tf.add_to_collection("pre256", tensor)

    with tf.variable_scope("enc_layer_2", reuse=tf.AUTO_REUSE):
      layer = tfc.SignalConv2D(
          num_filters, (5, 5), corr=True, strides_down=2, padding="same_zeros",
          use_bias=False, activation=None)
      tensor = layer(tensor)
      tf.add_to_collection("pre256", tensor)

    return tensor


def synthesis_transform(tensor, conv_filters_num, num_filters):
  """Builds the synthesis transform."""

  with tf.variable_scope("pre256", reuse=tf.AUTO_REUSE):
    with tf.variable_scope("dec_layer_0", reuse=tf.AUTO_REUSE):
      layer = tfc.SignalConv2D(
          conv_filters_num, (5, 5), corr=False, strides_up=2, padding="same_zeros",
          use_bias=True, activation=tfc.GDN(inverse=True))
      tensor = layer(tensor)
      tf.add_to_collection("pre256", tensor)

    with tf.variable_scope("dec_layer_1", reuse=tf.AUTO_REUSE):
      layer = tfc.SignalConv2D(
          conv_filters_num, (5, 5), corr=False, strides_up=2, padding="same_zeros",
          use_bias=True, activation=tfc.GDN(inverse=True))
      tensor = layer(tensor)
      tf.add_to_collection("pre256", tensor)

    with tf.variable_scope("dec_layer_2", reuse=tf.AUTO_REUSE):
      layer = tfc.SignalConv2D(
          3, (9, 9), corr=False, strides_up=4, padding="same_zeros",
          use_bias=True, activation=None)
      tensor = layer(tensor)
      tf.add_to_collection("pre256", tensor)

    return tensor


def kodak():
    imgs = np.zeros([24, wid, hgt, channels])
    pad = np.zeros([2])
    img_path = gb.glob("./kodak/*.png")
    pad[0] = 41 + 16 + 16 * np.ceil(wid / 16) - wid
    pad[1] = 41 + 16 + 16 * np.ceil(hgt / 16) - hgt
    idx = 0
    bdry = [np.floor(pad[0] / 2), np.floor(pad[1] / 2), np.ceil(pad[0] / 2), np.ceil(pad[1] / 2)]
    bdry = np.int32(bdry)
    for path in img_path:
        cur_img = mpimg.imread(path)
        cur_img = np.reshape(cur_img,[wid, hgt,channels])
        cur_img = np.float32(cur_img)
        imgs[idx, :, :, :] = cur_img[:, :, :]
        idx = idx + 1

    imgs_padded = np.pad(imgs, ((0,0), (bdry[0], bdry[2]), (bdry[1], bdry[3]), (0,0)), 'symmetric')
    return imgs, imgs_padded, bdry


def entropy_estimation(y, istrain):
  with tf.variable_scope("pre256", reuse=tf.AUTO_REUSE):
    entropy_bottleneck = tfc.EntropyBottleneck()
    y_tilde, likelihoods = entropy_bottleneck(y, training=istrain)
    if istrain:
      return y_tilde, likelihoods, entropy_bottleneck
    else:
      string = entropy_bottleneck.compress(y)
      string = tf.squeeze(string, axis=0)
      return y_tilde, likelihoods, entropy_bottleneck, string

def entropy_estimation_base(y, istrain):
  with tf.variable_scope("pre256", reuse=tf.AUTO_REUSE):
    entropy_bottleneck = tfc.EntropyBottleneck()
    y_tilde, likelihoods = entropy_bottleneck(y, training=istrain)
    string = entropy_bottleneck.compress(y)

    return y_tilde, likelihoods, entropy_bottleneck, string

def read_image(filename_queue):
    reader = tf.WholeFileReader("reader")
    key, record_string = reader.read(filename_queue)
    raw_img = tf.image.decode_image(record_string, channels=channels)
    crop_img = tf.random_crop(raw_img, [size, size, channels])
    float_img = tf.cast(crop_img, tf.float32) / 255.0
    disturb_img = float_img + tf.random_uniform([size, size, 1], minval=-0.001, maxval=0.001)

    return [disturb_img]

def input_pipeline(filename, batch_size, read_threads):
    filename_queue=tf.train.string_input_producer(filename, shuffle=True)
    img_list = [read_image(filename_queue) for _ in range(read_threads)]
    imgs=tf.reshape(img_list,[batch_size,size,size,channels])
    return imgs

def get_filename(dir_name):
    file = os.listdir(dir_name)
    filename = [os.path.join(dir_name, i_file) for i_file in file if i_file[-4:] == ".png"]
    return filename

def eval_test_image(sess, cur_data, string, img, y, eval_bpp, mse, num_pixels, x_hat, test_image, q):
    string_, x_shape, y_shape = sess.run([string, tf.shape(img), tf.shape(y)], feed_dict={img: cur_data})

    # Write a binary file with the shape information and the compressed string.
    with open("./kodak/" + str(q) + ".bin", "wb") as f:
        f.write(np.array(x_shape[1:-1], dtype=np.uint16).tobytes())
        f.write(np.array(y_shape[1:-1], dtype=np.uint16).tobytes())
        f.write(string_)

    # If requested, transform the quantized image back and measure performance.
    eval_bpp_, mse_, num_pixels_, x_hat_ = sess.run([eval_bpp, mse, num_pixels, x_hat], feed_dict={img: cur_data})

    x_hat_ = np.array(x_hat_)
    # psnr = 10 * np.log10(255 ** 2 / mse_)
    bpp = len(string_) * 8 / num_pixels_
    psnr_r = calc_psnr(np.squeeze(test_image[q, :, :, 0]), np.squeeze(x_hat_[:, :, :, 0]))
    psnr_g = calc_psnr(np.squeeze(test_image[q, :, :, 1]), np.squeeze(x_hat_[:, :, :, 1]))
    psnr_b = calc_psnr(np.squeeze(test_image[q, :, :, 2]), np.squeeze(x_hat_[:, :, :, 2]))
    psnr = (psnr_r + psnr_g + psnr_b) / 3

    msssim_r_rec = calc_ms_ssim_base(np.squeeze(test_image[q, :, :, 0]), np.squeeze(x_hat_[:, :, :, 0]))
    msssim_g_rec = calc_ms_ssim_base(np.squeeze(test_image[q, :, :, 1]), np.squeeze(x_hat_[:, :, :, 1]))
    msssim_b_rec = calc_ms_ssim_base(np.squeeze(test_image[q, :, :, 2]), np.squeeze(x_hat_[:, :, :, 2]))
    msssim_rec = (msssim_r_rec + msssim_g_rec + msssim_b_rec) / 3

    return psnr, bpp, x_hat_, msssim_rec

def eval_test_image_padded(sess, cur_data, string, img, y, eval_bpp, mse, num_pixels, x_hat, test_image, q, bdry):
    string_, x_shape, y_shape = sess.run([string, tf.shape(img), tf.shape(y)], feed_dict={img: cur_data})

    # Write a binary file with the shape information and the compressed string.
    with open("./kodak/" + str(q) + ".bin", "wb") as f:
        f.write(np.array(x_shape[1:-1], dtype=np.uint16).tobytes())
        f.write(np.array(y_shape[1:-1], dtype=np.uint16).tobytes())
        f.write(string_)

    # If requested, transform the quantized image back and measure performance.
    eval_bpp_, num_pixels_, x_hat_ = sess.run([eval_bpp, num_pixels, x_hat], feed_dict={img: cur_data})

    x_hat_ = np.array(x_hat_)
    # The actual bits per pixel including overhead.
    # psnr = 10 * np.log10(255 ** 2 / mse_)
    bpp = (8 + len(string_)) * 8 / num_pixels_
    psnr_r = calc_psnr(np.squeeze(test_image[q, bdry[0]:bdry[0]+wid, bdry[1]:bdry[1]+hgt, 0]), np.squeeze(x_hat_[:, bdry[0]:bdry[0]+wid, bdry[1]:bdry[1]+hgt, 0] / 255))
    psnr_g = calc_psnr(np.squeeze(test_image[q, bdry[0]:bdry[0]+wid, bdry[1]:bdry[1]+hgt, 1]), np.squeeze(x_hat_[:, bdry[0]:bdry[0]+wid, bdry[1]:bdry[1]+hgt, 1] / 255))
    psnr_b = calc_psnr(np.squeeze(test_image[q, bdry[0]:bdry[0]+wid, bdry[1]:bdry[1]+hgt, 2]), np.squeeze(x_hat_[:, bdry[0]:bdry[0]+wid, bdry[1]:bdry[1]+hgt, 2] / 255))
    psnr = (psnr_r + psnr_g + psnr_b) / 3
    x_hat_ = x_hat_[:, bdry[0]:bdry[0]+wid, bdry[1]:bdry[1]+hgt, :]

    return psnr, bpp, x_hat_
