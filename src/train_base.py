from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from util.util_base import *


def train():
  """Trains the model."""

  tf.logging.set_verbosity(tf.logging.INFO)

  filename = get_filename("./corpus/clic2018")
  images = input_pipeline(filename, args.batchsize, min(16, args.batchsize))

  # Training inputs are random crops out of the images tensor.
  crop_shape = (args.batchsize, args.patchsize, args.patchsize, 3)
  x = tf.random_crop(images, crop_shape)
  num_pixels = np.prod(crop_shape[:-1])

  # Build autoencoder.
  y = analysis_transform(x, args.conv_filters, args.num_filters)
  y_tilde, likelihoods, entropy_bottleneck = entropy_estimation(y, True)
  x_tilde = synthesis_transform(y_tilde, args.conv_filters, args.num_filters)

  # Total number of bits divided by number of pixels.
  train_bpp = tf.reduce_sum(tf.log(likelihoods)) / (-np.log(2) * num_pixels)

  # Mean squared error across pixels.
  train_mse = tf.reduce_sum(tf.squared_difference(x, x_tilde))
  # MS-SSIM: train_msssim = 1 - tf.reduce_mean(tf.image.ssim_multiscale(x, x_tilde, max_val=1.0))

  # Multiply by 255^2 to correct for rescaling.
  train_mse *= 255 ** 2 / num_pixels

  # The rate-distortion cost.
  train_loss =  train_mse + 1000 * train_bpp

  # Minimize loss and auxiliary loss, and execute update op.
  step = tf.train.create_global_step()
  main_optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
  main_step = main_optimizer.minimize(train_loss, global_step=step)

  aux_learning_rate = tf.train.exponential_decay(0.001, step, 5000, 0.96, staircase=False)
  aux_optimizer = tf.train.GradientDescentOptimizer(learning_rate=aux_learning_rate)
  aux_step = aux_optimizer.minimize(entropy_bottleneck.losses[0], global_step=step)

  train_op = tf.group(main_step, aux_step, entropy_bottleneck.updates[0])
  tf.add_to_collection("pre256", tf.trainable_variables())
  comp_vars_base = tf.trainable_variables()
  print(comp_vars_base)

  logged_tensors = [
      tf.identity(train_loss, name="train_loss"),
      tf.identity(train_bpp, name="train_bpp"),
      tf.identity(train_mse, name="train_mse"),
  ]
  hooks = [
      tf.train.StopAtStepHook(last_step=args.last_step),
      tf.train.NanTensorHook(train_loss),
      tf.train.LoggingTensorHook(logged_tensors, every_n_secs=8),
  ]

  with tf.train.MonitoredTrainingSession(
      hooks=hooks, checkpoint_dir=args.checkpoint_dir) as sess:
    while not sess.should_stop():
      sess.run(train_op)


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
      "--checkpoint_dir", default="color/base",
      help="Directory where to save/load model checkpoints.")
  parser.add_argument(
      "--batchsize", type=int, default=8,
      help="Batch size for training.")
  parser.add_argument(
      "--patchsize", type=int, default=256,
      help="Size of image patches for training.")
  parser.add_argument(
      "--lambda", type=float, default=0.1, dest="lmbda",
      help="Lambda for rate-distortion tradeoff.")
  parser.add_argument(
      "--last_step", type=int, default=100000,
      help="Train up to this number of steps.")

  args = parser.parse_args()

  train()

