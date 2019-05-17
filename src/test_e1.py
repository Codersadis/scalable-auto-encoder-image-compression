from util.util_error1 import *

def compress():
  """Compresses an image."""
  img = tf.placeholder(tf.float32, [1, None, None, 3])

  # Transform and compress the image, then remove batch dimension.
  y_base = analysis_transform(img, args.conv_filters, args.num_filters)
  y_hat, likelihoods, entropy_bottleneck, string = entropy_estimation(y_base, False)
  x_hat = synthesis_transform(y_hat,args.conv_filters, args.num_filters)
  num_pixels = tf.to_float(tf.reduce_prod(tf.shape(img)[:-1]))

  # e1 layer
  x_en = img - x_hat
  y_e1 = analysis_transform_e1(x_en, args.conv_filters, args.num_filters)
  y_hat_e1, likelihoods_e1, entropy_bottleneck_e1, string_e1 = entropy_estimation_e1(y_e1, False)
  x_hat_e1 = synthesis_transform_e1(y_hat_e1, args.conv_filters, args.num_filters)
  x_rec = x_hat_e1 + x_hat
  
  # Total number of bits divided by number of pixels.
  with tf.name_scope('rate'):
    eval_bpp    = tf.reduce_sum(tf.log(likelihoods)) / (-tf.log(2.0) * num_pixels)
    eval_bpp_e1 = tf.reduce_sum(tf.log(likelihoods_e1)) / (-tf.log(2.0) * num_pixels)

  # Mean squared error across pixels.
  x_hat = tf.clip_by_value(x_hat, 0, 1)

  comp_vars_base = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="pre256")
  comp_vars_e1 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="e1_pre256")
  comp_vars_e1 = [var for var in tf.global_variables() if var not in comp_vars_base]

  print(comp_vars_base)
  print(comp_vars_e1)

  test_image, test_image_padded, bdry = kodak()
  rec_image = np.zeros(np.shape(test_image))
  rec_image_e1_plus_base = np.zeros(np.shape(test_image))
  avg_entropy = []
  avg_psnr = []
  avg_entropy_e1 = []
  avg_psnr_e1_plus_base = []
  avg_ms_ssims = []

  ''' eval base layer '''
  with tf.Session() as sess:
    latest_base = tf.train.latest_checkpoint(checkpoint_dir=args.base_checkpoint_dir)
    latest_e1 = tf.train.latest_checkpoint(checkpoint_dir=args.e1_checkpoint_dir)
    tf.train.Saver(comp_vars_base).restore(sess, save_path=latest_base)
    tf.train.Saver(comp_vars_e1).restore(sess, save_path=latest_e1)
    for q in range(len(test_image)):
      cur_data = np.reshape(test_image[q, :, :, :], [1, wid, hgt, channels])
      psnr_rec, psnr_base, psnr_e1_plus_base, bpp_base, bpp_e1, x_hat_, x_hat_e1_plus_base_, y_hat_e1_, msssim_rec = eval_test_image_e1(sess,
                      cur_data, test_image, img, y_base, x_hat, x_rec, x_hat_e1, y_hat_e1, eval_bpp, string_e1, q)

      avg_entropy.append(bpp_base)
      avg_psnr.append(psnr_rec)
      avg_entropy_e1.append(bpp_e1)
      avg_psnr_e1_plus_base.append(psnr_e1_plus_base)
      avg_ms_ssims.append(msssim_rec)
      rec_image[q,:,:,:] = x_hat_
      rec_image_e1_plus_base[q,:,:,:] = x_hat_e1_plus_base_

      if (q % 8) == 0:
          print("Progress %d/%d." % (q, len(test_image)))

    base_avg_psnr = np.mean(avg_psnr)
    base_avg_entropy = np.mean(avg_entropy)
    psnr_e1_plus_base = np.mean(avg_psnr_e1_plus_base)
    e1_avg_entropy = np.mean(avg_entropy_e1)
    avg_msssim = np.mean(avg_ms_ssims)
    if os.path.exists("./color/base.png"):
      os.remove("./color/base.png")
    if os.path.exists("./color/base_plus_e1.png"):
      os.remove("./color/base_plus_e1.png")
    mpimg.imsave(r"./color/base.png", np.squeeze(rec_image[0,:,:,:]))
    mpimg.imsave(r"./color/base_plus_e1.png", np.squeeze(rec_image_e1_plus_base[0,:,:,:]))
    print('End-to-End PSNR: %.4f, E1+base PSNR: %.4f, Base bpp: %.4f, E1 bpp: %.4f final bpp: %.4f. ' %\
          (base_avg_psnr, psnr_e1_plus_base, base_avg_entropy, e1_avg_entropy, e1_avg_entropy + base_avg_entropy))
    print('End-to-End MS-SSIM: %.4f ' % (avg_msssim))

def decompress():
  """Decompresses an image."""

  # Read the shape information and compressed string from the binary file.
  with open(args.input, "rb") as f:
    x_shape = np.frombuffer(f.read(4), dtype=np.uint16)
    y_shape = np.frombuffer(f.read(4), dtype=np.uint16)
    string = f.read()

  y_shape = [int(s) for s in y_shape] + [args.num_filters]

  # Add a batch dimension, then decompress and transform the image back.
  strings = tf.expand_dims(string, 0)
  entropy_bottleneck = tfc.EntropyBottleneck(dtype=tf.float32)
  y_hat = entropy_bottleneck.decompress(
      strings, y_shape, channels=args.num_filters_en)
  x_hat = synthesis_transform(y_hat, args.num_filters)

  # or right boundaries.
  x_hat = x_hat[0, :x_shape[0], :x_shape[1], :]

  # Write reconstructed image out as a PNG file.
  op = save_image(args.output, x_rec)

  # Load the latest model checkpoint, and perform the above actions.
  with tf.Session() as sess:
    latest = tf.train.latest_checkpoint(checkpoint_dir=args.checkpoint_dir)
    tf.train.Saver().restore(sess, save_path=latest)
    sess.run(op)


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
      "--num_filters_en", type=int, default=48,
      help="Number of filters bottleneck layer of the enhance layer.")
	  
  parser.add_argument(
      "--e1_checkpoint_dir", default="color/model48-48-lambda3000-1000",
      help="Directory where to save/load e1 layer model checkpoints.")
  parser.add_argument(
      "--base_checkpoint_dir", default="color/model48-lambda3000",
      help="Directory where to save/load base layer model checkpoints.")

  args = parser.parse_args()

  compress()
