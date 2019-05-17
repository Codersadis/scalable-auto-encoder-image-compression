from util.util_base import *

def compress():
  """Compresses an image."""
  img = tf.placeholder(tf.float32, [1, None, None, 3])

  # Transform and compress the image, then remove batch dimension.
  y = analysis_transform(img, args.conv_filters, args.num_filters)
  y_hat, likelihoods, entropy_bottleneck, string = entropy_estimation(y, False)
  x_hat = synthesis_transform(y_hat,args.conv_filters, args.num_filters)
  num_pixels = tf.to_float(tf.reduce_prod(tf.shape(img)[:-1]))

  # Total number of bits divided by number of pixels.
  eval_bpp = tf.reduce_sum(tf.log(likelihoods)) / (-tf.log(2.0) * num_pixels)

  # Mean squared error across pixels.
  x_hat = tf.clip_by_value(x_hat, 0, 1)
  mse = tf.reduce_sum(tf.squared_difference(tf.round(img), x_hat)) / num_pixels

  base_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="pre256")
  print(base_var)
  test_image, test_image_padded, bdry = kodak()
  rec_image = np.zeros(np.shape(test_image))
  rec_image_padded = np.zeros(np.shape(test_image))
  avg_entropy = []
  avg_psnr = []
  avg_entropy_padded = []
  avg_psnr_padded = []
  avg_ms_ssims= []

  with tf.Session() as sess:
    # Load the latest model checkpoint, get the compressed string and the tensor
    # shapes.
    latest = tf.train.latest_checkpoint(checkpoint_dir=args.checkpoint_dir)
    tf.train.Saver().restore(sess, save_path=latest)
    for q in range(len(test_image)):
      cur_data = np.reshape(test_image[q, :, :, :], [1, wid, hgt, channels])
      psnr, bpp, x_hat_, msssim_rec = eval_test_image(sess, cur_data, string, img, y, eval_bpp, mse, num_pixels, x_hat, test_image, q)
      eval_bpp_ = sess.run([eval_bpp], feed_dict={img: cur_data})

      avg_entropy.append(bpp)
      avg_psnr.append(psnr)
      avg_ms_ssims.append(msssim_rec)
      rec_image[q,:,:,:] = x_hat_
      if (q % 8) == 0:
          print("Progress %d/%d." % (q, len(test_image)))
          print("Real bpp: %.4f, Eval bpp: %.4f" % (bpp, np.float32(eval_bpp_)))

    #print("Boundary padded test images")
    #for q in range(len(test_image_padded)):
    #  cur_data_padded = np.reshape(test_image_padded[q, :, :, :], [1, wid+bdry[0]+bdry[2], hgt+bdry[1]+bdry[3], channels])
    #  psnr, bpp, x_hat_ = eval_test_image_padded(sess, cur_data_padded, string, img, y, eval_bpp, mse, num_pixels, x_hat, test_image_padded, q, bdry)
    #  eval_bpp_ = sess.run([eval_bpp], feed_dict={img: cur_data_padded})
    #
    #  avg_entropy_padded.append(bpp)
    #  avg_psnr_padded.append(psnr)
    #  avg_psnr_padded.append(psnr)
    #  os.remove(r'./kodak/' + str(q) + '.bin')
    #  rec_image_padded[q,:,:,:] = x_hat_
    #  if (q % 8) == 0:
    #      print("Progress %d/%d." % (q, len(test_image_padded)))
    #      print("Real bpp: %.4f, Eval bpp: %.4f" % (bpp, np.float32(eval_bpp_)))

  final_avg_psnr = np.mean(avg_psnr)
  final_avg_entropy = np.mean(avg_entropy)
  avg_msssim = np.mean(avg_ms_ssims)
  #final_avg_psnr_padded = np.mean(avg_psnr_padded)
  #final_avg_entropy_padded = np.mean(avg_entropy_padded)
  if os.path.exists("./color/base.png"):
    os.remove("./color/base.png")
  mpimg.imsave(r"./color/base.png", np.squeeze(rec_image[0,:,:,:]))
  print('Base PSNR: %.4f, Coded Entropy: %.4f ' % (final_avg_psnr, final_avg_entropy))
  #print('Base PSNR: %.4f, Coded Entropy: %.4f (padded)' % (final_avg_psnr_padded, final_avg_entropy_padded))
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
      strings, y_shape, channels=args.num_filters)
  x_hat = synthesis_transform(y_hat, args.num_filters)

  # Remove batch dimension, and crop away any extraneous padding on the bottom
  # or right boundaries.
  x_hat = x_hat[0, :x_shape[0], :x_shape[1], :]

  # Write reconstructed image out as a PNG file.
  op = save_image(args.output, x_hat)

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
      "--checkpoint_dir", default="color/model48-lambda3000",
      help="Directory where to save/load model checkpoints.")

  args = parser.parse_args()

  compress()
