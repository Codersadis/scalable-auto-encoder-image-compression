from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from util.util_error1  import *
from util.util_error2 import *
from util.util_error3 import *

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
  y_tilde, likelihoods, entropy_bottleneck, string = entropy_estimation(y, False)
  x_tilde = synthesis_transform(y_tilde, args.conv_filters, args.num_filters)

  # Build enhance layer 1, 2, 3
  x_en = x - x_tilde
  y_e1 = analysis_transform_e1(x_en, args.conv_filters, args.num_filters)
  y_tilde_e1, likelihoods_e1, entropy_bottleneck_e1, string_e1 = entropy_estimation_e1(y_e1, False)
  x_tilde_e1 = synthesis_transform_e1(y_tilde_e1, args.conv_filters, args.num_filters)

  x_en2 = x - x_tilde - x_tilde_e1
  y_e2 = analysis_transform_e2(x_en2, args.conv_filters, args.num_filters * 2)
  y_tilde_e2, likelihoods_e2, entropy_bottleneck_e2, string_e2 = entropy_estimation_e2(y_e2, False)
  x_tilde_e2 = synthesis_transform_e2(y_tilde_e2, args.conv_filters, args.num_filters * 2)

  x_en3 = x - x_tilde - x_tilde_e1 - x_tilde_e2
  y_e3 = analysis_transform_e3(x_en3, args.conv_filters, args.num_filters * 2)
  y_tilde_e3, likelihoods_e3, entropy_bottleneck_e3 = entropy_estimation_e3(y_e3, True)
  x_tilde_e3 = synthesis_transform_e3(y_tilde_e3, args.conv_filters, args.num_filters * 2)
  x_rec = x_tilde + x_tilde_e1 + x_tilde_e2 + x_tilde_e3

  # Total number of bits divided by number of pixels.
  with tf.name_scope("rate"):
    train_bpp = tf.reduce_sum(tf.log(likelihoods)) / (-np.log(2) * num_pixels)
    train_bpp_e1 = tf.reduce_sum(tf.log(likelihoods_e1)) / (-np.log(2) * num_pixels)
    train_bpp_e2 = tf.reduce_sum(tf.log(likelihoods_e2)) / (-np.log(2) * num_pixels)
    train_bpp_e3 = tf.reduce_sum(tf.log(likelihoods_e3)) / (-np.log(2) * num_pixels)

  # Mean squared error across pixels.
  with tf.name_scope("distortion"):
    train_mse = tf.reduce_sum(tf.squared_difference(x, x_rec))
    train_mse *= 255 ** 2 / num_pixels # Multiply by 255^2 to correct for rescaling.

  # The rate-distortion cost.
  with tf.name_scope("rate_distortion_loss"):
    train_loss =  train_mse + args.lmbda * train_bpp_e3

  # collect variables
  comp_vars_base = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="pre256")
  comp_vars_e1   = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="e1_pre256")
  comp_vars_e2   = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="e2_pre256")
  prev_layer_vars = comp_vars_base + comp_vars_e1 + comp_vars_e2
  comp_vars_e3    = [var for var in tf.global_variables() if var not in prev_layer_vars]
  tf.add_to_collection('e3_pre256', comp_vars_e3)
  print(comp_vars_e3)

  # Minimize loss and auxiliary loss, and execute update op.
  step = tf.train.create_global_step()
  main_optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
  main_step = main_optimizer.minimize(train_loss, global_step=step,
                                      var_list=[var for var in comp_vars_e3])

  aux_learning_rate = tf.train.exponential_decay(0.1, step, 1000, 0.96, staircase=False)
  aux_optimizer = tf.train.AdamOptimizer(learning_rate=aux_learning_rate)
  aux_step = aux_optimizer.minimize(entropy_bottleneck_e3.losses[0], global_step=step,
                                    var_list=[var for var in comp_vars_e3])

  train_op = tf.group(main_step, aux_step, entropy_bottleneck_e3.updates[0])

  logged_tensors = [
      tf.identity(train_loss, name="train_loss"),
      tf.identity(train_bpp, name="train_bpp"),
      tf.identity(train_bpp_e1, name="train_bpp_e1"),
      tf.identity(train_bpp_e2, name="train_bpp_e2"),
      tf.identity(train_bpp_e3, name="train_bpp_e3"),
      tf.identity(train_mse, name="train_mse"),
  ]
  hooks = [
      tf.train.StopAtStepHook(last_step=args.last_step),
      tf.train.NanTensorHook(train_loss),
      tf.train.LoggingTensorHook(logged_tensors, every_n_secs=8),
  ]

  # partition the variables based on layer level
  saver_base = tf.train.Saver(comp_vars_base)
  saver_e1 = tf.train.Saver(comp_vars_e1)
  saver_e2 = tf.train.Saver(comp_vars_e2)
  saver_e3 = tf.train.Saver(comp_vars_e3)

  with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(args.base_checkpoint_dir)
    enhance_ckpt = tf.train.get_checkpoint_state(args.e1_checkpoint_dir)
    enhance2_ckpt = tf.train.get_checkpoint_state(args.e2_checkpoint_dir)
    enhance3_ckpt = tf.train.get_checkpoint_state(args.e3_checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      saver_base.restore(sess, tf.train.latest_checkpoint(args.base_checkpoint_dir))
      print("Restore the model from {}.".format(ckpt.model_checkpoint_path))
      if enhance_ckpt and enhance_ckpt.model_checkpoint_path:
        saver_e1.restore(sess, tf.train.latest_checkpoint(args.e1_checkpoint_dir))
        print("Restore the model from {}.".format(enhance_ckpt.model_checkpoint_path))
        if enhance2_ckpt and enhance2_ckpt.model_checkpoint_path:
          saver_e2.restore(sess, tf.train.latest_checkpoint(args.e2_checkpoint_dir))
          print("Restore the model from {}.".format(enhance2_ckpt.model_checkpoint_path))
          if enhance3_ckpt and enhance3_ckpt.model_checkpoint_path:
            saver_e3.restore(sess, tf.train.latest_checkpoint(args.e3_checkpoint_dir))
            print("Restore the model from {}.".format(enhance3_ckpt.model_checkpoint_path))

    with tf.train.MonitoredTrainingSession(
        hooks=hooks, checkpoint_dir=args.e3_checkpoint_dir)as sess:
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
      "--e3_checkpoint_dir", default="color/enhance3",
      help="Directory where to save/load e3 layer model checkpoints.")
  parser.add_argument(
      "--e2_checkpoint_dir", default="color/base3000_error1000_48-96_test300",
      help="Directory where to save/load e2 layer model checkpoints.")
  parser.add_argument(
      "--e1_checkpoint_dir", default="color/base3000_error1000_test",
      help="Directory where to save/load e1 layer model checkpoints.")
  parser.add_argument(
      "--base_checkpoint_dir", default="color/base3000",
      help="Directory where to save/load base layer model checkpoints.")
  parser.add_argument(
      "--batchsize", type=int, default=1,
      help="Batch size for training.")
  parser.add_argument(
      "--patchsize", type=int, default=256,
      help="Size of image patches for training.")
  parser.add_argument(
      "--lambda", type=float, default=200, dest="lmbda",
      help="Lambda for rate-distortion tradeoff.")
  parser.add_argument(
      "--last_step", type=int, default=50000,
      help="Train up to this number of steps.")

  args = parser.parse_args()

  train()
