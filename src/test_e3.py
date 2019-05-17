from util.util_error1 import *
from util.util_error2 import *
from util.util_error3 import *

def compress():
  """Compresses an image."""
  img = tf.placeholder(tf.float32, [1, None, None, 3])

  # Transform and compress the image, then remove batch dimension.
  y_base = analysis_transform(img, args.conv_filters, args.num_filters)
  y_hat, likelihoods, entropy_bottleneck, string = entropy_estimation(y_base, False)
  x_hat = synthesis_transform(y_hat,args.conv_filters, args.num_filters)
  num_pixels = tf.to_float(tf.reduce_prod(tf.shape(img)[:-1]))

  # e1 layer
  x_e1 = img - x_hat
  y_e1 = analysis_transform_e1(x_e1, args.conv_filters, args.num_filters)
  y_hat_e1, likelihoods_e1, entropy_bottleneck_e1, _ = entropy_estimation_e1(y_e1, False)
  x_hat_e1 = synthesis_transform_e1(y_hat_e1, args.conv_filters, args.num_filters)
  x_rec_e1 = x_hat +x_hat_e1

  # e2 layer
  x_e2 = img - x_hat - x_hat_e1
  y_e2 = analysis_transform_e2(x_e2, args.conv_filters, args.num_filters * 2)
  y_hat_e2, likelihoods_e2, entropy_bottleneck_e2, _ = entropy_estimation_e2(y_e2, False)
  x_hat_e2 = synthesis_transform_e2(y_hat_e2, args.conv_filters, args.num_filters * 2)

  # e3 layer
  x_e3 = img - x_hat - x_hat_e1 - x_hat_e2
  y_e3 = analysis_transform_e3(x_e3, args.conv_filters, args.num_filters * 3)
  y_hat_e3, likelihoods_e3, entropy_bottleneck_e3, string_e3 = entropy_estimation_e3(y_e3, False)
  x_hat_e3 = synthesis_transform_e3(y_hat_e3, args.conv_filters, args.num_filters * 3)

  x_rec = x_hat + x_hat_e1 + x_hat_e2 + x_hat_e3
  
  # Total number of bits divided by number of pixels.
  with tf.name_scope('rate'):
    eval_bpp    = tf.reduce_sum(tf.log(likelihoods)) / (-tf.log(2.0) * num_pixels)
    eval_bpp_e1 = tf.reduce_sum(tf.log(likelihoods_e1)) / (-tf.log(2.0) * num_pixels)
    eval_bpp_e2 = tf.reduce_sum(tf.log(likelihoods_e2)) / (-tf.log(2.0) * num_pixels)
    eval_bpp_e3 = tf.reduce_sum(tf.log(likelihoods_e3)) / (-tf.log(2.0) * num_pixels)

  # Mean squared error across pixels.
  x_hat = tf.clip_by_value(x_hat, 0, 1)

  comp_vars_base = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="pre256")
  comp_vars_e1 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="e1_pre256")
  comp_vars_e2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="e2_pre256")
  prev_layer_vars = comp_vars_base + comp_vars_e1 + comp_vars_e2
  comp_vars_e3    = [var for var in tf.global_variables() if var not in prev_layer_vars]
  print(comp_vars_e3)

  test_image, test_image_padded, bdry = kodak()
  rec_image = np.zeros(np.shape(test_image))
  rec_image_e1_plus_base = np.zeros(np.shape(test_image))
  rec_image_e2_plus_e1_plus_base = np.zeros(np.shape(test_image))
  rec_image_e3_e2_e1_base = np.zeros(np.shape(test_image))
  avg_psnr = []
  avg_psnr_e1_plus_base = []
  avg_entropy_e3 = []
  avg_psnr_e2_plus_e1_plus_base = []
  avg_psnr_rec = []  # e2e psnr
  avg_ms_ssims = []
  ''' eval base layer '''
  with tf.Session() as sess:
    latest_base = tf.train.latest_checkpoint(checkpoint_dir=args.base_checkpoint_dir)
    latest_e1 = tf.train.latest_checkpoint(checkpoint_dir=args.e1_checkpoint_dir)
    latest_e2 = tf.train.latest_checkpoint(checkpoint_dir=args.e2_checkpoint_dir)
    latest_e3 = tf.train.latest_checkpoint(checkpoint_dir=args.e3_checkpoint_dir)
    tf.train.Saver(comp_vars_base).restore(sess, save_path=latest_base)
    tf.train.Saver(comp_vars_e1).restore(sess, save_path=latest_e1)
    tf.train.Saver(comp_vars_e2).restore(sess, save_path=latest_e2)
    tf.train.Saver(comp_vars_e3).restore(sess, save_path=latest_e3)

    for q in range(len(test_image)):
      cur_data = np.reshape(test_image[q, :, :, :], [1, wid, hgt, channels])
      _, psnr_base, psnr_e1_plus_base, _, _, x_hat_, x_hat_e1_plus_base_, _, _ = eval_test_image_e1(sess,
                      cur_data, test_image, img, y_base,x_hat, x_rec_e1, x_hat_e1, y_hat_e1, eval_bpp, string_e3, q)

      _, psnr_e2_plus_e1_plus_base, _, x_hat_e2_plus_e1_plus_base_, _ = eval_test_image_e2(sess,
                      x_rec - x_hat_e3, cur_data, test_image, img, y_e2, x_hat_e2, x_hat_e1_plus_base_, string_e3, q)

      psnr_rec, psnr_e3_e2_e1_base, bpp_e3, x_hat_e3_e2_e1_base_, msssim_rec = eval_test_image_e3(sess,
                      x_rec, cur_data, test_image, img, y_e3, x_hat_e3, x_hat_e2_plus_e1_plus_base_, string_e3, q)

      avg_psnr.append(psnr_base)
      avg_psnr_e1_plus_base.append(psnr_e1_plus_base)
      avg_entropy_e3.append(bpp_e3)
      avg_psnr_e2_plus_e1_plus_base.append(psnr_e2_plus_e1_plus_base)
      avg_psnr_rec.append(psnr_rec)
      avg_ms_ssims.append(msssim_rec)
      rec_image[q,:,:,:] = x_hat_
      rec_image_e1_plus_base[q,:,:,:] = x_hat_e1_plus_base_
      rec_image_e2_plus_e1_plus_base[q,:,:,:] = x_hat_e2_plus_e1_plus_base_
      rec_image_e3_e2_e1_base[q,:,:,:] = x_hat_e3_e2_e1_base_

      if (q % 8) == 0:
          print("Progress %d/%d." % (q, len(test_image)))

    base_avg_psnr = np.mean(avg_psnr)
    psnr_e1_plus_base = np.mean(avg_psnr_e1_plus_base)
    psnr_e2_plus_e1_plus_base = np.mean(avg_psnr_e2_plus_e1_plus_base)
    rec_psnr = np.mean(avg_psnr_rec)
    e3_avg_entropy = np.mean(avg_entropy_e3)
    avg_msssim = np.mean(avg_ms_ssims)
    if os.path.exists("./color/base.png"):
      os.remove("./color/base.png")
    if os.path.exists("./color/base_e1.png"):
      os.remove("./color/base_e1.png")
    if os.path.exists("./color/base_e1_e2.png"):
      os.remove("./color/base_e1_e2.png")
    if os.path.exists("./color/base_e1_e2_e3.png"):
      os.remove("./color/base_e1_e2_e3.png")
    mpimg.imsave(r"./color/base.png", np.squeeze(rec_image[0,:,:,:]))
    mpimg.imsave(r"./color/base_e1.png", np.squeeze(rec_image_e1_plus_base[0,:,:,:]))
    mpimg.imsave(r"./color/base_e1_e2.png", np.squeeze(rec_image_e2_plus_e1_plus_base[0,:,:,:]))
    mpimg.imsave(r"./color/base_e1_e2_e3.png", np.squeeze(rec_image_e3_e2_e1_base[0,:,:,:]))
    print('End-to-End PSNR: %.4f, Base PSNR: %.4f, E1+base PSNR: %.4f, E2+E1+base: %.4f, E3 bpp: %.4f. ' %\
          (rec_psnr, base_avg_psnr, psnr_e1_plus_base, psnr_e2_plus_e1_plus_base, e3_avg_entropy))
    print('End-to-End MS-SSIM: %.4f ' % (avg_msssim))

if __name__ == "__main__":
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument(
      "--conv_filters", type=int, default=48,
      help="Number of filters conv layer.")
  parser.add_argument(
      "--num_filters", type=int, default=48,
      help="Number of filters bottleneck layer.")
  parser.add_argument(
      "--e3_checkpoint_dir", default="color/model48-48-96-144-lambda3000-1000-300-100",
      help="Directory where to save/load e3 layer model checkpoints.")
  parser.add_argument(
      "--e2_checkpoint_dir", default="color/model48-48-96-lambda3000-1000-300",
      help="Directory where to save/load e2 layer model checkpoints.")
  parser.add_argument(
      "--e1_checkpoint_dir", default="color/model48-48-lambda3000-1000",
      help="Directory where to save/load e1 layer model checkpoints.")
  parser.add_argument(
      "--base_checkpoint_dir", default="color/model48-lambda3000",
      help="Directory where to save/load base layer model checkpoints.")

  args = parser.parse_args()

  compress()
