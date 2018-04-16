import numpy as np
import os
import tensorflow as tf
import time
import bisect


def weight_variable(shape):
    # Xavier initialization
    stddev = np.sqrt(2.0 / (sum(shape)))
    initial = tf.truncated_normal(shape, stddev=stddev)
    weights = tf.Variable(initial)
    tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, weights)
    return weights


def bias_variable(shape, trainable=True):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial, trainable=trainable)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, data_format='NCHW', strides=[1, 1, 1, 1], padding='SAME')


class NeuralNet:
    def __init__(self):
        self.root_dir = './saved_models/'

        # Network structure
        self.residual_filter = 64
        self.residual_blocks = 6

        # For exporting
        self.weights = []
        self.session = tf.Session()
        self.saver = tf.train.Saver()

        self.training = tf.placeholder(tf.bool)
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.batch_norm_count = 0

        # Training
        self.policy_loss_weight = 1.0
        self.value_loss_weight = 1.0
        self.learning_rate_boundaries = [100000, 130000]
        self.learning_rate_values = [0.0005, 0.002, 0.02]
        self.learning_rate = self.learning_rate_values[0]
        self.num_steps_train = 200
        self.num_steps_test = 2000
        self.total_steps = 140000

        # Summary
        self.test_writer = tf.summary.FileWriter(os.path.join(os.getcwd(), "logs/test/"), self.session.graph)
        self.train_writer = tf.summary.FileWriter(os.path.join(os.getcwd(), "logs/train/"), self.session.graph)

        # Other variables to be defined upon init
        self.handle = None
        self.next_batch = None
        self.train_handle = None
        self.test_handle = None
        self.x = None
        self.y_ = None
        self.z_ = None
        self.y_conv = None
        self.z_conv = None
        self.policy_loss = None
        self.mse_loss = None
        self.reg_term = None
        self.update_ops = None
        self.train_op = None
        self.accuracy = None
        self.avg_policy_loss = []
        self.avg_mse_loss = []
        self.avg_reg_term = None
        self.time_start = None

    def construct(self, dataset, train_iterator, test_iterator):
        # TF variables
        self.handle = tf.placeholder(tf.string, shape=[])
        self.next_batch = tf.data.Iterator.from_string_handle(self.handle, dataset.output_types, dataset.output_shapes).get_next()
        self.train_handle = self.session.run(train_iterator.string_handle())
        self.test_handle = self.session.run(test_iterator.string_handle())

        self.x = self.next_batch[0]   # tf.placeholder(tf.float32, [None, 112, 8*8])
        self.y_ = self.next_batch[1]  # tf.placeholder(tf.float32, [None, 1858])
        self.z_ = self.next_batch[2]  # tf.placeholder(tf.float32, [None, 1])
        self.y_conv, self.z_conv = self.construct_net(self.x)

        # Calculate loss on policy head
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=self.y_conv)
        self.policy_loss = tf.reduce_mean(cross_entropy)

        # Loss on value head
        self.mse_loss = tf.reduce_mean(tf.squared_difference(self.z_, self.z_conv))

        # Regularizer
        regularizer = tf.contrib.layers.l2_regularizer(scale=0.0001)
        reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.reg_term = tf.contrib.layers.apply_regularization(regularizer, reg_variables)

        # For training from a (smaller) dataset of strong players, you will
        # want to reduce the factor in front of self.mse_loss here.
        loss = self.policy_loss_weight * self.policy_loss + self.value_loss_weight * self.mse_loss + self.reg_term

        # You need to change the learning rate here if you are training
        # from a self-play training set, for example start with 0.005 instead.
        opt_op = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.9, use_nesterov=True)

        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(self.update_ops):
            self.train_op = opt_op.minimize(loss, global_step=self.global_step)

        correct_prediction = tf.equal(tf.argmax(self.y_conv, 1), tf.argmax(self.y_, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
        self.accuracy = tf.reduce_mean(correct_prediction)

        self.session.run(tf.global_variables_initializer())

    def replace_weights(self, new_weights):
        for e, weights in enumerate(self.weights):
            # Keyed batchnorm weights
            if isinstance(weights, str):
                work_weights = tf.get_default_graph().get_tensor_by_name(weights)
                new_weight = tf.constant(new_weights[e])
                self.session.run(tf.assign(work_weights, new_weight))
            elif weights.shape.ndims == 4:
                # Transpose convolutation weights from [filter_height, filter_width, in_channels, out_channels]
                # to [output, input, filter_size, filter_size]
                s = weights.shape.as_list()
                shape = [s[i] for i in [3, 2, 0, 1]]
                new_weight = tf.constant(new_weights[e], shape=shape)
                self.session.run(weights.assign(tf.transpose(new_weight, [2, 3, 1, 0])))
            elif weights.shape.ndims == 2:
                # Change fully connected layers from [in, out] to [out, in]
                s = weights.shape.as_list()
                shape = [s[i] for i in [1, 0]]
                new_weight = tf.constant(new_weights[e], shape=shape)
                self.session.run(weights.assign(tf.transpose(new_weight, [1, 0])))
            else:
                # Biases, batchnorm etc
                new_weight = tf.constant(new_weights[e], shape=weights.shape)
                self.session.run(weights.assign(new_weight))

    def restore(self, file):
        print(f'Restoring from {file}')
        self.saver.restore(self.session, file)

    def process(self, batch_size, test_batches):
        if not self.time_start:
            self.time_start = time.time()

        # Run training for this batch
        policy_loss, mse_loss, reg_term, _, _ = self.session.run(
            [self.policy_loss, self.mse_loss, self.reg_term, self.train_op, self.next_batch],
            feed_dict={self.training: True, self.learning_rate: self.learning_rate, self.handle: self.train_handle}
        )

        steps = tf.train.global_step(self.session, self.global_step)

        # Determine learning rate
        steps_total = (steps-1) % self.total_steps
        self.learning_rate = self.learning_rate_values[bisect.bisect_right(self.learning_rate_boundaries, steps_total)]

        # Keep running averages
        # Google's paper scales MSE by 1/4 to a [0, 1] range, so do the same to
        # get comparable values.
        mse_loss /= 4.0
        self.avg_policy_loss.append(policy_loss)
        self.avg_mse_loss.append(mse_loss)
        self.avg_reg_term.append(reg_term)
        if steps % self.num_steps_train == 0:
            pol_loss_w = self.policy_loss_weight
            val_loss_w = self.value_loss_weight
            time_end = time.time()
            speed = 0
            if self.time_start:
                elapsed = time_end - self.time_start
                speed = batch_size * (self.num_steps_train / elapsed)
            avg_policy_loss = np.mean(self.avg_policy_loss or [0])
            avg_mse_loss = np.mean(self.avg_mse_loss or [0])
            avg_reg_term = np.mean(self.avg_reg_term or [0])
            print(f'Steps {steps},'
                  f'lr={self.learning_rate}'
                  f'policy={avg_policy_loss}'
                  f'mse={avg_mse_loss}'
                  f'reg={avg_reg_term}'
                  f'total={pol_loss_w * avg_policy_loss + val_loss_w * avg_mse_loss + avg_reg_term}'
                  f'({speed} pos/s)'
            )
            train_summaries = tf.Summary(value=[
                tf.Summary.Value(tag="Policy Loss", simple_value=avg_policy_loss),
                tf.Summary.Value(tag="MSE Loss", simple_value=avg_mse_loss)])
            self.train_writer.add_summary(train_summaries, steps)
            self.time_start = time_end
            self.avg_policy_loss, self.avg_mse_loss, self.avg_reg_term = [], [], []

        if steps % self.num_steps_test == 0:
            sum_accuracy = 0
            sum_mse = 0
            sum_policy = 0
            for _ in range(0, test_batches):
                test_policy, test_accuracy, test_mse, _ = self.session.run(
                    [self.policy_loss, self.accuracy, self.mse_loss,
                     self.next_batch],
                    feed_dict={self.training: False,
                               self.handle: self.test_handle})
                sum_accuracy += test_accuracy
                sum_mse += test_mse
                sum_policy += test_policy
            sum_accuracy /= test_batches
            sum_accuracy *= 100
            sum_policy /= test_batches
            # Additionally rescale to [0, 1] so divide by 4
            sum_mse /= (4.0 * test_batches)
            test_summaries = tf.Summary(value=[
                tf.Summary.Value(tag="Accuracy", simple_value=sum_accuracy),
                tf.Summary.Value(tag="Policy Loss", simple_value=sum_policy),
                tf.Summary.Value(tag="MSE Loss", simple_value=sum_mse)])
            self.test_writer.add_summary(test_summaries, steps)
            print(f'Steps {steps}, policy={sum_policy} training accuracy={sum_accuracy}%, mse={sum_mse}')
            save_path = self.saver.save(self.session, self.root_dir, global_step=steps)
            print("Model saved in file: {}".format(save_path))
            weights_path = self.root_dir + "-" + str(steps) + ".txt"
            self.save_weights(weights_path)
            print(f'Weights saved in file: {weights_path}')

    def save_weights(self, filename):
        with open(filename, "w") as f:
            for weights in self.weights:
                f.write("\n")
                # Keyed batchnorm weights
                if isinstance(weights, str):
                    work_weights = tf.get_default_graph().get_tensor_by_name(weights)
                elif weights.shape.ndims == 4:
                    # Transpose convolution weights [filter_height, filter_width, in_channels, out_channels]
                    # to [output, input, filter_size, filter_size]
                    work_weights = tf.transpose(weights, [3, 2, 0, 1])
                elif weights.shape.ndims == 2:
                    # Change fully connected layers from [in, out] (TF) to [out, in]
                    work_weights = tf.transpose(weights, [1, 0])
                else:
                    # Biases, batchnorm etc
                    work_weights = weights
                wt_str = [str(wt) for wt in np.ravel(work_weights.eval(session=self.session))]
                f.write(" ".join(wt_str))

    def get_batchnorm_key(self):
        result = "bn" + str(self.batch_norm_count)
        self.batch_norm_count += 1
        return result

    def conv_block(self, inputs, filter_size, input_channels, output_channels):
        W_conv = weight_variable([filter_size, filter_size,
                                  input_channels, output_channels])
        b_conv = bias_variable([output_channels], False)
        self.weights.append(W_conv)
        self.weights.append(b_conv)

        weight_key = self.get_batchnorm_key()
        self.weights.append(weight_key + "/batch_normalization/moving_mean:0")
        self.weights.append(weight_key + "/batch_normalization/moving_variance:0")

        with tf.variable_scope(weight_key):
            h_bn = \
                tf.layers.batch_normalization(
                    conv2d(inputs, W_conv),
                    epsilon=1e-5, axis=1, fused=True,
                    center=False, scale=False,
                    training=self.training)
        h_conv = tf.nn.relu(h_bn)
        return h_conv

    def residual_block(self, inputs, channels):
        # First convnet
        orig = tf.identity(inputs)
        W_conv_1 = weight_variable([3, 3, channels, channels])
        b_conv_1 = bias_variable([channels], False)
        self.weights.append(W_conv_1)
        self.weights.append(b_conv_1)
        weight_key_1 = self.get_batchnorm_key()
        self.weights.append(weight_key_1 + "/batch_normalization/moving_mean:0")
        self.weights.append(weight_key_1 + "/batch_normalization/moving_variance:0")

        # Second convnet
        W_conv_2 = weight_variable([3, 3, channels, channels])
        b_conv_2 = bias_variable([channels], False)
        self.weights.append(W_conv_2)
        self.weights.append(b_conv_2)
        weight_key_2 = self.get_batchnorm_key()
        self.weights.append(weight_key_2 + "/batch_normalization/moving_mean:0")
        self.weights.append(weight_key_2 + "/batch_normalization/moving_variance:0")

        with tf.variable_scope(weight_key_1):
            h_bn1 = \
                tf.layers.batch_normalization(
                    conv2d(inputs, W_conv_1),
                    epsilon=1e-5, axis=1, fused=True,
                    center=False, scale=False,
                    training=self.training)
        h_out_1 = tf.nn.relu(h_bn1)
        with tf.variable_scope(weight_key_2):
            h_bn2 = \
                tf.layers.batch_normalization(
                    conv2d(h_out_1, W_conv_2),
                    epsilon=1e-5, axis=1, fused=True,
                    center=False, scale=False,
                    training=self.training)
        h_out_2 = tf.nn.relu(tf.add(h_bn2, orig))
        return h_out_2

    def construct_net(self, planes):
        # NCHW format
        # batch, 112 input channels, 8 x 8
        x_planes = tf.reshape(planes, [-1, 112, 8, 8])

        # Input convolution
        flow = self.conv_block(x_planes, filter_size=3,
                               input_channels=112,
                               output_channels=self.residual_filter)
        # Residual tower
        for _ in range(self.residual_blocks):
            flow = self.residual_block(flow, self.residual_filter)

        # Policy head
        conv_pol = self.conv_block(flow, filter_size=1,
                                   input_channels=self.residual_filter,
                                   output_channels=32)
        h_conv_pol_flat = tf.reshape(conv_pol, [-1, 32*8*8])
        W_fc1 = weight_variable([32*8*8, 1858])
        b_fc1 = bias_variable([1858])
        self.weights.append(W_fc1)
        self.weights.append(b_fc1)
        h_fc1 = tf.add(tf.matmul(h_conv_pol_flat, W_fc1), b_fc1, name='policy_head')

        # Value head
        conv_val = self.conv_block(flow, filter_size=1, input_channels=self.residual_filter, output_channels=32)
        h_conv_val_flat = tf.reshape(conv_val, [-1, 32*8*8])
        W_fc2 = weight_variable([32 * 8 * 8, 128])
        b_fc2 = bias_variable([128])
        self.weights.append(W_fc2)
        self.weights.append(b_fc2)
        h_fc2 = tf.nn.relu(tf.add(tf.matmul(h_conv_val_flat, W_fc2), b_fc2))
        W_fc3 = weight_variable([128, 1])
        b_fc3 = bias_variable([1])
        self.weights.append(W_fc3)
        self.weights.append(b_fc3)
        h_fc3 = tf.nn.tanh(tf.add(tf.matmul(h_fc2, W_fc3), b_fc3), name='value_head')

        return h_fc1, h_fc3
