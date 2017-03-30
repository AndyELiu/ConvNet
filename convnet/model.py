import tensorflow as tf
import os
import math
import numpy as np
from .layers import build_network


class ConvNet(object):
    def __init__(self, config):
        self.config = config
        working_dir = config['working_dir']
        working_dir = os.path.expandvars(working_dir)
        self.working_dir = os.path.expanduser(working_dir)
        self.model_chkt_path = os.path.join(self.working_dir, 'model.chkt')
        self.image_config = config['image']
        self.model_config = config['model']
        self.train_config = config['train']
        self.preprocessing_config = config['preprocessing']
        self.eval_config = config['eval']

        # build graph
        self._build_train_model()
        self._build_eval_model()

        # model saver and summary writer
        self.sess = tf.Session()
        self.saver = tf.train.Saver(tf.global_variables())
        self.summary_writer = tf.summary.FileWriter(
            self.working_dir, self.sess.graph)

        self.summary_op = tf.summary.merge_all()
        self.sess.run(tf.global_variables_initializer())
        tf.train.start_queue_runners(sess=self.sess)

    def _build_train_model(self):
        train_data_dir = self.image_config['train_data_dir']
        expand_dir = os.path.expandvars(train_data_dir)
        expand_dir = os.path.expanduser(expand_dir)
        train_data_files = self.image_config['train_data_files']
        filelists = [os.path.join(expand_dir, fname)
                     for fname in train_data_files]

        images, labels = self._build_image_pipeline(filelists, 'train')
        logits = self._build_layers(images)

        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels,
            logits=logits))

        tf.add_to_collection('losses', loss)
        self.loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
        self.train_op = self._build_optimizer()

        self.model_save_interval = self.train_config.get('model_save_interval',
                                                         100)

        tf.summary.scalar('loss', self.loss)

    def _build_eval_model(self):
        eval_data_dir = self.image_config['eval_data_dir']
        expand_dir = os.path.expandvars(eval_data_dir)
        eval_data_files = self.image_config['eval_data_files']
        filelists = [os.path.join(expand_dir, fname)
                     for fname in eval_data_files]

        images, labels = self._build_image_pipeline(filelists, 'eval')
        logits = self._build_layers(images, reuse=True)

        self.top_k_op = tf.nn.in_top_k(logits, labels, 1)

        self.model_eval_interval = self.eval_config['model_eval_interval']

    def _build_image_pipeline(self, filelists, mode='train'):
        for f in filelists:
            if not tf.gfile.Exists(f):
                raise ValueError('Failed to find file: ' + f)

        label_bytes = self.image_config['label_bytes']
        height = self.image_config['height']
        width = self.image_config['width']
        depth = self.image_config['depth']
        image_bytes = height * width * depth
        record_bytes = label_bytes + image_bytes

        # Create a queue that produces the filenames to read.
        filename_queue = tf.train.string_input_producer(filelists)
        reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
        key, value = reader.read(filename_queue)

        record_bytes = tf.decode_raw(value, tf.uint8)

        label = tf.cast(
            tf.strided_slice(record_bytes, [0], [label_bytes]), tf.int32)

        depth_major = tf.reshape(
            tf.strided_slice(record_bytes, [label_bytes],
                             [label_bytes + image_bytes]), [depth, height,
                                                            width])
        image = tf.transpose(depth_major, [1, 2, 0])

        image = tf.cast(image, tf.float32)

        if 'crop' in self.preprocessing_config:
            height = self.preprocessing_config['crop']['height']
            width = self.preprocessing_config['crop']['width']
            if mode == 'train':
                image = tf.random_crop(image, [height, width, 3])
            else:
                image = tf.image.resize_image_with_crop_or_pad(image, height,
                                                               width)

        if mode == 'train':
            # Randomly flip the image horizontally.
            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_brightness(image, max_delta=63)
            image = tf.image.random_contrast(image, lower=0.2, upper=1.8)

        # Subtract off the mean and divide by the variance of the pixels.
        image = tf.image.per_image_standardization(image)

        # Set the shapes of tensors.
        image.set_shape([height, width, 3])
        label.set_shape([1])

        # Ensure that the random shuffling has good mixing properties.
        min_fraction_of_examples_in_queue = 0.4
        if mode == 'train':
            num_examples = self.image_config['num_train_examples']
        else:
            num_examples = self.image_config['num_eval_examples']
        min_queue_examples = int(
            num_examples * min_fraction_of_examples_in_queue)
        print('Filling queue with {} CIFAR images for {}'.format(
            min_queue_examples, mode))

        num_preprocess_threads = 16
        batch_size = self.train_config['batch_size']
        if mode == 'train':
            images, label_batch = tf.train.shuffle_batch(
                [image, label],
                batch_size=batch_size,
                num_threads=num_preprocess_threads,
                capacity=min_queue_examples + 3 * batch_size,
                min_after_dequeue=min_queue_examples)
        else:
            images, label_batch = tf.train.batch(
                [image, label],
                batch_size=batch_size,
                num_threads=num_preprocess_threads,
                capacity=min_queue_examples + 3 * batch_size)

        return images, tf.reshape(label_batch, [batch_size])

    def _build_layers(self, images, reuse=False):
        # logits and loss
        network = self.model_config['network']
        layer_defs = self.model_config['layer_defs']
        logits = build_network(images, network, layer_defs, reuse)
        return logits

    def _build_optimizer(self):
        train_config = self.train_config
        optimizer_str = train_config.get('optimizer', 'GradientDescent')
        start_learning_rate = train_config.get('learning_rate', 0.001)
        decay_steps = train_config.get('decay_steps', 1000)
        decay_rate = train_config.get('decay_rate', 0.98)

        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        learning_rate = tf.train.exponential_decay(
            start_learning_rate,
            self.global_step,
            decay_steps,
            decay_rate,
            staircase=True)

        tf.summary.scalar('learning_rate', learning_rate)
        if optimizer_str == 'GradientDescent':
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        elif optimizer_str == 'AdaDelta':
            optimizer = tf.train.AdadeltaOptimizer(learning_rate)
        elif optimizer_str == 'AdaGrid':
            optimizer = tf.train.AdagridOptimizer(learning_rate)
        elif optimizer_str == 'Adam':
            optimizer = tf.train.AdamOptimizer(learning_rate)
        else:
            optimizer = tf.train.RMSPropOptimizer(learning_rate)
        train_op = optimizer.minimize(self.loss, global_step=self.global_step)
        return train_op

    def _save_model(self, save_path):
        print('saving model ....')
        self.saver.save(self.sess, save_path)

    def _load_model(self, load_path):
        print('loading model ...')
        self.saver.restore(self.sess, load_path)

    def train(self, max_steps=100000):
        for step in range(max_steps):
            self.step = step
            _, summary, loss = self.sess.run(
                [self.train_op, self.summary_op, self.loss])
            print("setps: {}, loss: {}".format(step, loss))
            self.summary_writer.add_summary(summary, step)
            if step % self.model_save_interval == 0:
                self._save_model(self.model_chkt_path)
            if step % self.model_eval_interval == 0:
                self._eval()

    def _add_eval_summary(self, accuracy):
        summary = tf.Summary(value=[
            tf.Summary.Value(tag='eval_accuracy',
                             simple_value=accuracy)
        ])
        self.summary_writer.add_summary(summary, self.step)

    def _eval(self):
        batch_size = self.train_config['batch_size']
        num_eval_examples = self.image_config['num_eval_examples']
        num_iter = int(math.ceil(num_eval_examples / batch_size))

        true_count = 0  # Counts the number of correct predictions.
        total_sample_count = num_iter * batch_size
        for _ in range(num_iter):
            predictions = self.sess.run([self.top_k_op])
            true_count += np.sum(predictions)

        # Compute precision @ 1.
        precision = true_count / total_sample_count
        print('precision @ 1 = {}'.format(precision))

        self._add_eval_summary(precision)
