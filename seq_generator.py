from sklearn import metrics
import numpy as np
import tensorflow as tf
from tensorflow.python.util import nest
import random


class SeqGenerator(object):
    """
    The basic class for generator
    """
    def train(self, X, sequence_length):
        """ train a generator according to X
        :param X: the data matrix. shape: [num examples, max sequence length].
                  or the list of data matrix(first element is malware, second element is benign)
        :param sequence_length: the length of each row of X. shape: [num examples]
                  or the list of length matrix(first element is malware, second element is benign)
        """
        raise NotImplementedError("Abstract method")

    def sample(self, X, sequence_length):
        """ generate samples for X
        :param X: the data matrix. shape: [num examples, max sequence length].
                  or the list of data matrix(first element is malware, second element is benign)
        :param sequence_length: the length of each row of X. shape: [num examples]
                  or the list of length matrix(first element is malware, second element is benign)
        :return: the generated samples Y, the corresponding sequence_length
        """
        raise NotImplementedError("Abstract method")


class MergerCell():
    """The cell used to merge two sequences"""

    def __init__(self, cell, inputs_0, inputs_0_length, inputs_1, inputs_1_length,
                 input_noise=False, output_cell_output=False):
        """

        :param cell: the RNN cell used to merge the two sequences
        :param inputs_0: the first sequence. shape [batch_size, max sequence length, dim]
        :param inputs_0_length: the length of the first sequence. shape [batch_size]
        :param inputs_1: the second sequence. shape [batch_size, max sequence length, dim]
        :param inputs_1_length: the length of the second sequence. shape [batch_size]
        :param self.input_noise: whether use noise as inputs
        :param output_cell_output: whether to concat the inner cell's output to MergerCell's output
        """
        self.cell = cell
        self.batch_size = inputs_0.get_shape()[0].value
        self.max_length = inputs_0.get_shape()[1].value
        self.inputs_0 = tf.reshape(inputs_0, [-1, inputs_0.get_shape()[2].value])
        self.inputs_0_length = inputs_0_length
        self.inputs_1 = tf.reshape(inputs_1, [-1, inputs_1.get_shape()[2].value])
        self.inputs_1_length = inputs_1_length
        self.input_noise = input_noise
        self.output_cell_output = output_cell_output

    @property
    def state_size(self):
        if isinstance(self.cell.state_size, tuple):
            return (1, 1) + self.cell.state_size
        else:
            return tuple(1, 1, self.cell.state_size)


    @property
    def output_size(self):
        if self.output_cell_output:
            return 1 + self.cell.output_size
        else:
            return 1

    def zero_state(self, batch_size, dtype):
        """ Return zero-filled state tensors
        :param batch_size: the batch size
        :param dtype: the data type to use for the cell state
        :return: a tuple of states
        """
        inputs_0_index = tf.zeros([batch_size, 1], dtype=tf.float32)
        inputs_1_index = tf.zeros([batch_size, 1], dtype=tf.float32)

        if nest.is_sequence(self.cell.state_size):
            state_size_flat = nest.flatten(self.cell.state_size)
            zeros_flat = [tf.zeros((batch_size, s), dtype=tf.float32) for s in state_size_flat]
            cell_state = nest.pack_sequence_as(structure=self.cell.state_size, flat_sequence=zeros_flat)
        else:
            cell_state = tf.zeros((batch_size, self.cell.state_size), dtype=tf.float32)

        #cell_state = self.cell.zero_state(batch_size, dtype)
        if isinstance(self.cell.state_size, tuple):
            state = (inputs_0_index, inputs_1_index) + cell_state
        else:
            state = (inputs_0_index, inputs_1_index, cell_state)
        return state


    def __call__(self, inputs, state, scope=None):
        """ merge one element
        :param inputs: if input_noise is True, inputs is the noise, otherwise not used.
                       Other input are integrated in state. shape [batch_size, dim]
        :param state: tuple of state Tensors, both `2-D`,
                      the first two are the input index in inputs_[0|1], shape [batch size, 1]
                      following are state of cell
        :param scope: VariableScope for the created subgraph; defaults to "merger_cell".
        :return: a tuple containing: output probability of the second sequence with shape [batch size, 1], the state
        """
        with tf.variable_scope(scope or "merger_cell"):
            inputs_0_index = tf.to_int32(state[0])
            inputs_1_index = tf.to_int32(state[1])
            if isinstance(self.cell.state_size, tuple):
                cell_state = state[2:]
            else:
                cell_state = state[2]
            indices = tf.range(self.batch_size) * self.max_length + tf.squeeze(inputs_0_index)
            inputs_0_content = tf.gather(self.inputs_0, indices)
            indices = tf.range(self.batch_size) * self.max_length + tf.squeeze(inputs_1_index)
            inputs_1_content = tf.gather(self.inputs_1, indices)
            inputs_content = tf.concat([inputs_0_content, inputs_1_content], 1)
            if self.input_noise:
                inputs_content = tf.concat([inputs_content, inputs], 1)

            cell_output, cell_state = self.cell(inputs_content, cell_state)
            output = tf.nn.rnn_cell._linear(cell_output, 1, bias=True, scope="output_projection")
            output = tf.nn.sigmoid(output)

            predicted_sequence = tf.to_int32(tf.greater_equal(output, 0.5))
            inputs_0_index += 1 - predicted_sequence
            inputs_0_index = tf.to_float(tf.minimum(inputs_0_index, tf.expand_dims(self.inputs_0_length, 1) - 1))
            inputs_1_index += predicted_sequence
            inputs_1_index = tf.to_float(tf.minimum(inputs_1_index, tf.expand_dims(self.inputs_1_length, 1) - 1))

            if isinstance(self.cell.state_size, tuple):
                state = (inputs_0_index, inputs_1_index) + cell_state
            else:
                state = (inputs_0_index, inputs_1_index, cell_state)

            if self.output_cell_output:
                output = tf.concat([output, cell_output], 1)
        return output, state


class SeqMerger(SeqGenerator):
    """
    Merge two sequences
    """
    def __init__(self, D, cell_type='LSTM', encoder_layers=[512], encoder_bidirectional=True,
                 encoder_share_weights=True, merge_layers=[512], batch_size=128, noise_dim=16,
                 num_tokens=161, max_length=2048, max_epoch=1000, max_epoch_no_improvement=25,
                 num_samples=1, learning_rate=0.001, baseline=True, model_path='model'):
        """
        :param cell_type: 'LSTM', 'RNN', 'GRU'
        :param encoder_layers: a list of integer
        :param encoder_bidirectional: whether to use bidirectional RNN for encoder
        :param encoder_share_weights: whether the two sequences use the same weights of RNN encoder
        :param merge_layers: a list of integer
        :param batch_size: the size of mini-batch
        :param noise_dim: the dimension of noise input. <=0 means not using noise
        :param num_tokens: the number of distinct tokens of input data
        :param max_length: the maximum length of the input sequence
        :param max_epoch: the maximum epoch for training RNN
        :param max_epoch_no_improvement: if the performance of the val set doesn't improve for
                max_epoch_no_improvement, stop training
        :param num_samples: the number of samples used for Policy Gradients, must divide exactly batch_size
        :param learning_rate: the learning rate for RNN
        :param baseline: whether to use baseline for the rewards of RL
        :param model_path: the path to save the model
        """
        self.D = D
        self.cell_type = cell_type
        self.encoder_layers = encoder_layers
        self.encoder_bidirectional = encoder_bidirectional
        self.encoder_share_weights = encoder_share_weights
        self.merge_layers = merge_layers
        self.batch_size = batch_size
        self.noise_dim = noise_dim
        self.num_tokens = num_tokens
        self.max_length = max_length
        self.max_epoch = max_epoch
        self.max_epoch_no_improvement = max_epoch_no_improvement
        self.num_samples = num_samples
        self.learning_rate = learning_rate
        self.baseline = baseline
        self.model_path = model_path
        g = tf.Graph()
        with g.as_default():
            self._build_model()
            self.saver = tf.train.Saver(max_to_keep=0)
        self.sess = tf.Session(graph=g)

    def _build_model(self):
        # initializing the cell type
        if self.cell_type is 'RNN':
            cell_element = tf.nn.rnn_cell.BasicRNNCell
        elif self.cell_type is 'LSTM':
            cell_element = tf.nn.rnn_cell.BasicLSTMCell
        elif self.cell_type is 'GRU':
            cell_element = tf.nn.rnn_cell.GRUCell
        else:
            raise ValueError('cell_type must be one of "LSTM", "RNN", "GRU"')

        # set the depth of encoder cell
        if len(self.encoder_layers) == 1:
            cell = cell_element(self.encoder_layers[0])
        elif len(self.encoder_layers) > 1:
            cell_elements = []
            for encoder_layer in self.encoder_layers:
                cell_elements.append(cell_element(encoder_layer))
            cell = tf.nn.rnn_cell.MultiRNNCell(cell_elements)

        # inputs
        self.inputs_0 = tf.placeholder(tf.int32, [self.batch_size, self.max_length])
        self.inputs_0_length = tf.placeholder(tf.int32, [self.batch_size])
        self.inputs_1 = tf.placeholder(tf.int32, [self.batch_size, self.max_length])
        self.inputs_1_length = tf.placeholder(tf.int32, [self.batch_size])
        self.noise_inputs = tf.placeholder(tf.float32, [self.batch_size, 2 * self.max_length,
                                                        self.noise_dim if self.noise_dim > 0 else 1])
        noise_inputs_length = tf.add(self.inputs_0_length, self.inputs_1_length)
        inputs_0_one_hot = tf.one_hot(self.inputs_0, self.num_tokens)
        inputs_1_one_hot = tf.one_hot(self.inputs_1, self.num_tokens)

        # encoder
        if len(self.encoder_layers) > 0:
            with tf.variable_scope("Encoder"):
                # bidirectional
                if self.encoder_bidirectional:
                    merger_inputs_0, _ = tf.nn.bidirectional_dynamic_rnn(cell, cell, inputs_0_one_hot,
                                                                         self.inputs_0_length, dtype=tf.float32,
                                                                         swap_memory=True, time_major=False,
                                                                         scope='bidirectional_rnn_0')
                    merger_inputs_0 = tf.concat(merger_inputs_0, 2)
                    if self.encoder_share_weights:
                        tf.get_variable_scope().reuse_variables()
                        encoder_scope = 'bidirectional_rnn_0'
                    else:
                        encoder_scope = 'bidirectional_rnn_1'
                    merger_inputs_1, _ = tf.nn.bidirectional_dynamic_rnn(cell, cell, inputs_1_one_hot,
                                                                         self.inputs_1_length, dtype=tf.float32,
                                                                         swap_memory=True, time_major=False,
                                                                         scope=encoder_scope)
                    merger_inputs_1 = tf.concat(merger_inputs_1, 2)

                else:
                    merger_inputs_0, _ = tf.nn.dynamic_rnn(cell, inputs_0_one_hot, self.inputs_0_length,
                                                           dtype=tf.float32, swap_memory=True, time_major=False,
                                                           scope='rnn_0')
                    if self.encoder_share_weights:
                        tf.get_variable_scope().reuse_variables()
                        encoder_scope = 'rnn_0'
                    else:
                        encoder_scope = 'rnn_1'
                    merger_inputs_1, _ = tf.nn.dynamic_rnn(cell, inputs_1_one_hot, self.inputs_1_length,
                                                           dtype=tf.float32, swap_memory=True, time_major=False,
                                                           scope=encoder_scope)

        else:
            merger_inputs_0 = inputs_0_one_hot
            merger_inputs_1 = inputs_1_one_hot

        # merger
        # set the depth of merger cell
        if len(self.merge_layers) == 1:
            cell = cell_element(self.merge_layers[0])
        elif len(self.merge_layers) > 1:
            cell_elements = []
            for merge_layer in self.merge_layers:
                cell_elements.append(cell_element(merge_layer))
            cell = tf.nn.rnn_cell.MultiRNNCell(cell_elements)
        input_noise = True if self.noise_dim > 0 else False
        merge_cell = MergerCell(cell, merger_inputs_0, self.inputs_0_length, merger_inputs_1, self.inputs_1_length,
                                input_noise)
        with tf.variable_scope("Merger"):
            initial_state = merge_cell.zero_state(self.batch_size, tf.float32)
            merger_outputs, _ = tf.nn.dynamic_rnn(merge_cell, self.noise_inputs, noise_inputs_length,
                                                  initial_state=initial_state,
                                                  swap_memory=True, time_major=False)

        probability = tf.reshape(merger_outputs, [self.batch_size, 2 * self.max_length])
        self.predicted_sequence = tf.to_int32(tf.greater_equal(probability, 0.5))

        self.predicted_sequence_valid_length = tf.placeholder(tf.int32, [self.batch_size])
        self.discriminator_result = tf.placeholder(tf.int32, [self.batch_size])
        self.loss = -((tf.to_float(self.predicted_sequence) *
                     tf.log(tf.clip_by_value(probability, 1e-10, 1.0))
                     + (1 - tf.to_float(self.predicted_sequence)) *
                     tf.log(tf.clip_by_value(1 - probability, 1e-10, 1.0))))
        predicted_sequence_mask = tf.sequence_mask(self.predicted_sequence_valid_length,
                                                   2 * self.max_length, dtype=tf.float32)
        self.loss *= predicted_sequence_mask
        self.loss = tf.reduce_sum(self.loss, axis=1)
        # debug
        self.loss_0 = tf.reduce_sum(self.loss * tf.to_float(1 - self.discriminator_result)) \
                      / (tf.reduce_sum(tf.to_float(1 - self.discriminator_result)) + 1e-10)
        self.loss_1 = tf.reduce_sum(self.loss * tf.to_float(self.discriminator_result)) \
                      / (tf.reduce_sum(tf.to_float(self.discriminator_result)) + 1e-10)
        # end of debug
        if self.baseline:
            rewards = tf.to_float(1 - 2 * self.discriminator_result)
            mean_rewards = tf.reduce_mean(
                tf.reshape(rewards, [self.batch_size / self.num_samples, self.num_samples]),
                axis=1, keep_dims=True)
            mean_rewards = tf.reshape(tf.tile(mean_rewards, [1, self.num_samples]), [self.batch_size])
            self.loss *= rewards - mean_rewards
        else:
            self.loss *= tf.to_float(1 - 2 * self.discriminator_result)
        self.loss = tf.reduce_sum(self.loss) / self.batch_size

        opt = tf.train.AdamOptimizer(self.learning_rate)
        grads_and_vars = opt.compute_gradients(self.loss)
        grads_and_vars = [(tf.clip_by_value(gv[0], -1.0, 1.0), gv[1]) for gv in grads_and_vars]
        self.train_op = opt.apply_gradients(grads_and_vars)
        self.init_op = tf.global_variables_initializer()

    def train(self, X, sequence_length):
        """ train a generator according to X
        :param X: the list of data matrix(first element is malware, second element is benign)
                  the data matrix shape: [num examples, max sequence length].
        :param sequence_length: the list of length (first element is malware, second element is benign)
                  the length of each row of X. shape: [num examples]
        """
        self.sess.run(self.init_op)
        # get the benign data
        X_benign = X[1]
        X = X[0]
        benign_sequence_length = sequence_length[1]
        sequence_length = sequence_length[0]

        # shuffle and split train and val data
        index = np.arange(len(X))
        np.random.shuffle(index)
        X = X[index]
        sequence_length = sequence_length[index]
        num_training_samples = int(len(X) * 0.75)
        X_val = X[num_training_samples:]
        sequence_length_val = sequence_length[num_training_samples:]
        X = X[:num_training_samples]
        sequence_length = sequence_length[:num_training_samples]

        best_val_tpr = 1.0
        best_val_epoch = 0
        for epoch in range(self.max_epoch):
            train_loss = 0.0
            train_loss_0 = 0.0
            train_loss_1 = 0.0
            train_tpr = 0.0
            distinct_batch_size = self.batch_size / self.num_samples
            for start, end in zip(range(0, len(X), distinct_batch_size),
                                  range(distinct_batch_size, len(X) + 1, distinct_batch_size)):
                X_batch = np.repeat(X[start: end], self.num_samples, axis=0)
                sequence_length_batch = np.repeat(sequence_length[start: end], self.num_samples, axis=0)
                benign_index = np.random.random_integers(0, len(X_benign) - 1, self.batch_size)
                X_benign_batch = X_benign[benign_index]
                benign_sequence_length_batch = benign_sequence_length[benign_index]
                noise_inputs_batch = np.random.rand(self.batch_size, 2 * self.max_length,
                                                    self.noise_dim if self.noise_dim > 0 else 1)
                # merge the two sequence
                predicted_sequence_batch = self.sess.run(self.predicted_sequence, feed_dict={
                    self.inputs_0: X_benign_batch,
                    self.inputs_0_length: benign_sequence_length_batch,
                    self.inputs_1: X_batch,
                    self.inputs_1_length: sequence_length_batch,
                    self.noise_inputs: noise_inputs_batch
                })
                # get the merged sequence
                predicted_sequence_valid_length_batch = np.zeros((self.batch_size,), dtype=np.int32)
                merged_batch = np.zeros((self.batch_size, 2 * self.max_length), dtype=np.int32)
                merged_sequence_length_batch = np.zeros((self.batch_size,), dtype=np.int32)
                for i in range(self.batch_size):
                    merged_sequence_length_batch[i] = benign_sequence_length_batch[i] + sequence_length_batch[i]
                    idx_0 = 0
                    idx_1 = 0
                    for j in range(2 * self.max_length):
                        if predicted_sequence_batch[i, j] == 0:
                            merged_batch[i, j] = X_benign_batch[i, idx_0]
                            idx_0 += 1
                            if idx_0 >= benign_sequence_length_batch[i]:
                                predicted_sequence_valid_length_batch[i] = j + 1
                                remaining_length = sequence_length_batch[i] - idx_1
                                merged_batch[i, j + 1: j + 1 + remaining_length] = \
                                    X_batch[i, idx_1: idx_1 + remaining_length]
                                break
                        else:
                            merged_batch[i, j] = X_batch[i, idx_1]
                            idx_1 += 1
                            if idx_1 >= sequence_length_batch[i]:
                                predicted_sequence_valid_length_batch[i] = j + 1
                                remaining_length = benign_sequence_length_batch[i] - idx_0
                                merged_batch[i, j + 1: j + 1 + remaining_length] = \
                                    X_benign_batch[i, idx_0: idx_0 + remaining_length]
                                break

                discriminator_result_batch = self.D.predict(merged_batch, merged_sequence_length_batch)

                _, loss_value, l0, l1 = self.sess.run([self.train_op, self.loss, self.loss_0, self.loss_1], feed_dict={
                    self.inputs_0: X_benign_batch,
                    self.inputs_0_length: benign_sequence_length_batch,
                    self.inputs_1: X_batch,
                    self.inputs_1_length: sequence_length_batch,
                    self.noise_inputs: noise_inputs_batch,
                    self.predicted_sequence_valid_length: predicted_sequence_valid_length_batch,
                    self.discriminator_result: discriminator_result_batch
                })
                train_loss += loss_value
                train_loss_0 += l0
                train_loss_1 += l1
                train_tpr += discriminator_result_batch.mean()
            train_loss /= len(X) / distinct_batch_size
            train_loss_0 /= len(X) / distinct_batch_size
            train_loss_1 /= len(X) / distinct_batch_size
            train_tpr /= len(X) / distinct_batch_size
            self.saver.save(self.sess, self.model_path, global_step=epoch)

            val_loss = 0.0
            val_loss_0 = 0.0
            val_loss_1 = 0.0
            val_tpr = 0.0
            for start, end in zip(range(0, len(X_val), distinct_batch_size),
                                  range(distinct_batch_size, len(X_val) + 1, distinct_batch_size)):
                X_batch = np.repeat(X_val[start: end], self.num_samples, axis=0)
                sequence_length_batch = np.repeat(sequence_length_val[start: end], self.num_samples, axis=0)
                benign_index = np.random.random_integers(0, len(X_benign) - 1, self.batch_size)
                X_benign_batch = X_benign[benign_index]
                benign_sequence_length_batch = benign_sequence_length[benign_index]
                noise_inputs_batch = np.random.rand(self.batch_size, 2 * self.max_length,
                                                    self.noise_dim if self.noise_dim > 0 else 1)
                # merge the two sequence
                predicted_sequence_batch = self.sess.run(self.predicted_sequence, feed_dict={
                    self.inputs_0: X_benign_batch,
                    self.inputs_0_length: benign_sequence_length_batch,
                    self.inputs_1: X_batch,
                    self.inputs_1_length: sequence_length_batch,
                    self.noise_inputs: noise_inputs_batch
                })
                # get the merged sequence
                predicted_sequence_valid_length_batch = np.zeros((self.batch_size,), dtype=np.int32)
                merged_batch = np.zeros((self.batch_size, 2 * self.max_length), dtype=np.int32)
                merged_sequence_length_batch = np.zeros((self.batch_size,), dtype=np.int32)
                for i in range(self.batch_size):
                    merged_sequence_length_batch[i] = benign_sequence_length_batch[i] + sequence_length_batch[i]
                    idx_0 = 0
                    idx_1 = 0
                    for j in range(2 * self.max_length):
                        if predicted_sequence_batch[i, j] == 0:
                            merged_batch[i, j] = X_benign_batch[i, idx_0]
                            idx_0 += 1
                            if idx_0 >= benign_sequence_length_batch[i]:
                                predicted_sequence_valid_length_batch[i] = j + 1
                                remaining_length = sequence_length_batch[i] - idx_1
                                merged_batch[i, j + 1: j + 1 + remaining_length] = \
                                    X_batch[i, idx_1: idx_1 + remaining_length]
                                break
                        else:
                            merged_batch[i, j] = X_batch[i, idx_1]
                            idx_1 += 1
                            if idx_1 >= sequence_length_batch[i]:
                                predicted_sequence_valid_length_batch[i] = j + 1
                                remaining_length = benign_sequence_length_batch[i] - idx_0
                                merged_batch[i, j + 1: j + 1 + remaining_length] = \
                                    X_benign_batch[i, idx_0: idx_0 + remaining_length]
                                break

                discriminator_result_batch = self.D.predict(merged_batch, merged_sequence_length_batch)

                loss_value, l0, l1 = self.sess.run([self.loss, self.loss_0, self.loss_1], feed_dict={
                    self.inputs_0: X_benign_batch,
                    self.inputs_0_length: benign_sequence_length_batch,
                    self.inputs_1: X_batch,
                    self.inputs_1_length: sequence_length_batch,
                    self.noise_inputs: noise_inputs_batch,
                    self.predicted_sequence_valid_length: predicted_sequence_valid_length_batch,
                    self.discriminator_result: discriminator_result_batch
                })
                val_loss += loss_value
                val_loss_0 += l0
                val_loss_1 += l1
                val_tpr += discriminator_result_batch.mean()
            val_loss /= len(X_val) / distinct_batch_size
            val_loss_0 /= len(X_val) / distinct_batch_size
            val_loss_1 /= len(X_val) / distinct_batch_size
            val_tpr /= len(X_val) / distinct_batch_size

            if val_tpr < best_val_tpr:
                best_val_tpr = val_tpr
                best_val_epoch = epoch

            log_message = 'Generator Epoch %d: Loss / TPR: Training=%g(%g,%g) / %g, Val=%g(%g,%g) / %g' % \
                          (epoch, train_loss, train_loss_0, train_loss_1, train_tpr,
                           val_loss, val_loss_0, val_loss_1, val_tpr)
            print(log_message)
            with open(self.model_path + '--training_log.txt', 'a') as f:
                f.write(log_message + '\n')

            if epoch - best_val_epoch >= self.max_epoch_no_improvement:
                self.saver.restore(self.sess, self.model_path + '-' + str(best_val_epoch))
                break

    def sample(self, X, sequence_length):
        """ generate samples for X
        :param X: the list of data matrix(first element is malware, second element is benign)
                  the data matrix shape: [num examples, max sequence length].
        :param sequence_length: the list of length (first element is malware, second element is benign)
                  the length of each row of X. shape: [num examples]
        :return: the generated samples, the corresponding sequence_length
        """
        # get the benign data
        X_benign = X[1]
        X = X[0]
        benign_sequence_length = sequence_length[1]
        sequence_length = sequence_length[0]

        num_samples = len(X)
        X = np.concatenate((X, X[:self.batch_size - 1]))
        sequence_length = np.concatenate((sequence_length, sequence_length[:self.batch_size - 1]))
        generated_X = np.zeros((len(X), 2 * self.max_length), dtype=np.int32)
        generated_sequence_length = np.zeros((len(X),), dtype=np.int32)

        for start, end in zip(range(0, len(X), self.batch_size),
                                range(self.batch_size, len(X) + 1, self.batch_size)):
            X_batch = X[start: end]
            sequence_length_batch = sequence_length[start: end]
            benign_index = np.random.random_integers(0, len(X_benign) - 1, self.batch_size)
            X_benign_batch = X_benign[benign_index]
            benign_sequence_length_batch = benign_sequence_length[benign_index]
            noise_inputs_batch = np.random.rand(self.batch_size, 2 * self.max_length,
                                                self.noise_dim if self.noise_dim > 0 else 1)
            # merge the two sequence
            predicted_sequence_batch = self.sess.run(self.predicted_sequence, feed_dict={
                self.inputs_0: X_benign_batch,
                self.inputs_0_length: benign_sequence_length_batch,
                self.inputs_1: X_batch,
                self.inputs_1_length: sequence_length_batch,
                self.noise_inputs: noise_inputs_batch
            })
            # get the merged sequence
            merged_batch = np.zeros((self.batch_size, 2 * self.max_length), dtype=np.int32)
            merged_sequence_length_batch = np.zeros((self.batch_size,), dtype=np.int32)
            for i in range(self.batch_size):
                merged_sequence_length_batch[i] = benign_sequence_length_batch[i] + sequence_length_batch[i]
                idx_0 = 0
                idx_1 = 0
                for j in range(2 * self.max_length):
                    if predicted_sequence_batch[i, j] == 0:
                        merged_batch[i, j] = X_benign_batch[i, idx_0]
                        idx_0 += 1
                        if idx_0 >= benign_sequence_length_batch[i]:
                            remaining_length = sequence_length_batch[i] - idx_1
                            merged_batch[i, j + 1: j + 1 + remaining_length] = \
                                X_batch[i, idx_1: idx_1 + remaining_length]
                            break
                    else:
                        merged_batch[i, j] = X_batch[i, idx_1]
                        idx_1 += 1
                        if idx_1 >= sequence_length_batch[i]:
                            remaining_length = benign_sequence_length_batch[i] - idx_0
                            merged_batch[i, j + 1: j + 1 + remaining_length] = \
                                X_benign_batch[i, idx_0: idx_0 + remaining_length]
                            break
            generated_X[start: end] = merged_batch
            generated_sequence_length[start: end] = merged_sequence_length_batch

        return generated_X[:num_samples], generated_sequence_length[:num_samples]


class SeqMergerA3C(SeqGenerator):
    """
    Merge two sequences using advantage actor critic
    """
    def __init__(self, D, cell_type='LSTM', encoder_layers=[512], encoder_bidirectional=True,
                 encoder_share_weights=True, merge_layers=[512], value_layers=[512], batch_size=128, gamma=0.99,
                 eps=0.01, noise_dim=16, num_tokens=161, max_length=2048, max_epoch=1000, max_epoch_no_improvement=25,
                 learning_rate=0.001, model_path='model'):
        """
        :param cell_type: 'LSTM', 'RNN', 'GRU'
        :param encoder_layers: a list of integer
        :param encoder_bidirectional: whether to use bidirectional RNN for encoder
        :param encoder_share_weights: whether the two sequences use the same weights of RNN encoder
        :param merge_layers: a list of integer
        :param value_layers: the layers of the value network, a list of integer
        :param batch_size: the size of mini-batch
        :param gamma: gamma for value function
        :param eps: eps reward for keep state
        :param noise_dim: the dimension of noise input. <=0 means not using noise
        :param num_tokens: the number of distinct tokens of input data
        :param max_length: the maximum length of the input sequence
        :param max_epoch: the maximum epoch for training RNN
        :param max_epoch_no_improvement: if the performance of the val set doesn't improve for
                max_epoch_no_improvement, stop training
        :param learning_rate: the learning rate for RNN
        :param model_path: the path to save the model
        """
        self.D = D
        self.cell_type = cell_type
        self.encoder_layers = encoder_layers
        self.encoder_bidirectional = encoder_bidirectional
        self.encoder_share_weights = encoder_share_weights
        self.merge_layers = merge_layers
        self.value_layers = value_layers + [1]
        self.batch_size = batch_size
        self.gamma = gamma
        self.eps = eps
        self.noise_dim = noise_dim
        self.num_tokens = num_tokens
        self.max_length = max_length
        self.max_epoch = max_epoch
        self.max_epoch_no_improvement = max_epoch_no_improvement
        self.learning_rate = learning_rate
        self.model_path = model_path
        g = tf.Graph()
        with g.as_default():
            self._build_model()
            self.saver = tf.train.Saver(max_to_keep=0)
        self.sess = tf.Session(graph=g)

    def _build_model(self):
        # initializing the cell type
        if self.cell_type is 'RNN':
            cell_element = tf.nn.rnn_cell.BasicRNNCell
        elif self.cell_type is 'LSTM':
            cell_element = tf.nn.rnn_cell.BasicLSTMCell
        elif self.cell_type is 'GRU':
            cell_element = tf.nn.rnn_cell.GRUCell
        else:
            raise ValueError('cell_type must be one of "LSTM", "RNN", "GRU"')

        # set the depth of encoder cell
        if len(self.encoder_layers) == 1:
            cell = cell_element(self.encoder_layers[0])
        elif len(self.encoder_layers) > 1:
            cell_elements = []
            for encoder_layer in self.encoder_layers:
                cell_elements.append(cell_element(encoder_layer))
            cell = tf.nn.rnn_cell.MultiRNNCell(cell_elements)

        # inputs
        self.inputs_0 = tf.placeholder(tf.int32, [self.batch_size, self.max_length])
        self.inputs_0_length = tf.placeholder(tf.int32, [self.batch_size])
        self.inputs_1 = tf.placeholder(tf.int32, [self.batch_size, self.max_length])
        self.inputs_1_length = tf.placeholder(tf.int32, [self.batch_size])
        self.noise_inputs = tf.placeholder(tf.float32, [self.batch_size, 2 * self.max_length,
                                                        self.noise_dim if self.noise_dim > 0 else 1])
        noise_inputs_length = tf.add(self.inputs_0_length, self.inputs_1_length)
        inputs_0_one_hot = tf.one_hot(self.inputs_0, self.num_tokens)
        inputs_1_one_hot = tf.one_hot(self.inputs_1, self.num_tokens)

        # encoder
        if len(self.encoder_layers) > 0:
            with tf.variable_scope("Encoder"):
                # bidirectional
                if self.encoder_bidirectional:
                    merger_inputs_0, _ = tf.nn.bidirectional_dynamic_rnn(cell, cell, inputs_0_one_hot,
                                                                         self.inputs_0_length, dtype=tf.float32,
                                                                         swap_memory=True, time_major=False,
                                                                         scope='bidirectional_rnn_0')
                    merger_inputs_0 = tf.concat(merger_inputs_0, 2)
                    if self.encoder_share_weights:
                        tf.get_variable_scope().reuse_variables()
                        encoder_scope = 'bidirectional_rnn_0'
                    else:
                        encoder_scope = 'bidirectional_rnn_1'
                    merger_inputs_1, _ = tf.nn.bidirectional_dynamic_rnn(cell, cell, inputs_1_one_hot,
                                                                         self.inputs_1_length, dtype=tf.float32,
                                                                         swap_memory=True, time_major=False,
                                                                         scope=encoder_scope)
                    merger_inputs_1 = tf.concat(merger_inputs_1, 2)

                else:
                    merger_inputs_0, _ = tf.nn.dynamic_rnn(cell, inputs_0_one_hot, self.inputs_0_length,
                                                           dtype=tf.float32, swap_memory=True, time_major=False,
                                                           scope='rnn_0')
                    if self.encoder_share_weights:
                        tf.get_variable_scope().reuse_variables()
                        encoder_scope = 'rnn_0'
                    else:
                        encoder_scope = 'rnn_1'
                    merger_inputs_1, _ = tf.nn.dynamic_rnn(cell, inputs_1_one_hot, self.inputs_1_length,
                                                           dtype=tf.float32, swap_memory=True, time_major=False,
                                                           scope=encoder_scope)

        else:
            merger_inputs_0 = inputs_0_one_hot
            merger_inputs_1 = inputs_1_one_hot

        # merger
        # set the depth of merger cell
        if len(self.merge_layers) == 1:
            cell = cell_element(self.merge_layers[0])
        elif len(self.merge_layers) > 1:
            cell_elements = []
            for merge_layer in self.merge_layers:
                cell_elements.append(cell_element(merge_layer))
            cell = tf.nn.rnn_cell.MultiRNNCell(cell_elements)
        input_noise = True if self.noise_dim > 0 else False
        merge_cell = MergerCell(cell, merger_inputs_0, self.inputs_0_length, merger_inputs_1, self.inputs_1_length,
                                input_noise, output_cell_output=True)
        with tf.variable_scope("Merger"):
            initial_state = merge_cell.zero_state(self.batch_size, tf.float32)
            merger_outputs, _ = tf.nn.dynamic_rnn(merge_cell, self.noise_inputs, noise_inputs_length,
                                                  initial_state=initial_state,
                                                  swap_memory=True, time_major=False)

        probability = tf.reshape(merger_outputs[:, :, 0], [self.batch_size, 2 * self.max_length])
        self.predicted_sequence = tf.to_int32(tf.greater_equal(probability, 0.5))

        self.predicted_sequence_valid_length = tf.placeholder(tf.int32, [self.batch_size])
        self.discriminator_per_step_result = tf.placeholder(tf.int32, [self.batch_size, 2 * self.max_length])
        predicted_sequence_mask = tf.sequence_mask(self.predicted_sequence_valid_length,
                                                   2 * self.max_length, dtype=tf.float32)

        # the value network
        value_hidden = tf.reshape(merger_outputs[:, :, 1:],
                                  [self.batch_size * 2 * self.max_length, self.merge_layers[-1]])
        last_value_hidden = self.merge_layers[-1]
        params_value = []
        for layer, value_layer in enumerate(self.value_layers):
            W_value = tf.Variable(tf.random_uniform([last_value_hidden, value_layer], -0.1, 0.1),
                               name='W_value_%d' % (layer,))
            last_value_hidden = value_layer
            b_value = tf.Variable(tf.zeros([value_layer]), name='b_value_%d' % (layer,))
            params_value.extend([W_value, b_value])
            value_hidden = tf.nn.tanh(tf.matmul(value_hidden, W_value) + b_value)
        value_output = tf.reshape(value_hidden, [self.batch_size, 2 * self.max_length])
        value_output = tf.concat([value_output, tf.zeros([self.batch_size, 1])], 1)
        zero_time_reward = tf.to_float(1 - 2 * self.discriminator_per_step_result[:, :1])
        nonzero_time_reward = self.discriminator_per_step_result[:, :-1] - self.discriminator_per_step_result[:, 1:]
        eps_rewards = tf.to_int32(tf.equal(nonzero_time_reward, 0))\
                      * (1 - 2 * self.discriminator_per_step_result[:, 1:])
        eps_rewards = self.eps * tf.to_float(eps_rewards)
        nonzero_time_reward = tf.to_float(nonzero_time_reward) + eps_rewards
        reward = tf.concat([zero_time_reward, nonzero_time_reward], 1)
        reward = tf.reverse_sequence(reward, noise_inputs_length, seq_dim=1, batch_dim=0)
        reward_list = [reward[:, 0]]
        for i in range(1, 2 * self.max_length):
            reward_list.append(reward[:, i] + reward_list[i - 1] * self.gamma)
        reward = tf.pack(reward_list, axis=1)
        reward = tf.reverse_sequence(reward, noise_inputs_length, seq_dim=1, batch_dim=0)
        advantage = reward - value_output[:, :-1]

        # policy loss
        self.loss = -((tf.to_float(self.predicted_sequence) *
                     tf.log(tf.clip_by_value(probability, 1e-10, 1.0))
                     + (1 - tf.to_float(self.predicted_sequence)) *
                     tf.log(tf.clip_by_value(1 - probability, 1e-10, 1.0))))
        self.loss *= tf.stop_gradient(advantage)
        self.loss *= predicted_sequence_mask
        self.loss = tf.reduce_sum(self.loss) / self.batch_size
        opt = tf.train.AdamOptimizer(self.learning_rate)
        grads_and_vars = opt.compute_gradients(self.loss)
        grads_and_vars = [(tf.clip_by_value(gv[0], -1.0, 1.0) if gv[0] is not None else gv[0], gv[1])
                          for gv in grads_and_vars]
        self.train_op = opt.apply_gradients(grads_and_vars)

        # value loss
        self.value_loss = value_output[:, :-1] - tf.stop_gradient(reward)
        self.value_loss *= self.value_loss
        self.value_loss *= predicted_sequence_mask
        self.value_loss = tf.reduce_sum(self.value_loss) / self.batch_size
        value_opt = tf.train.AdamOptimizer(self.learning_rate)
        value_grads_and_vars = value_opt.compute_gradients(self.value_loss, params_value)
        value_grads_and_vars = [(tf.clip_by_value(gv[0], -1.0, 1.0), gv[1]) for gv in value_grads_and_vars]
        self.value_train_op = value_opt.apply_gradients(value_grads_and_vars)

        # supervised loss
        self.supervised_sequence = tf.placeholder(tf.int32, [self.batch_size, 2 * self.max_length])
        self.supervised_sequence_valid_length = tf.placeholder(tf.int32, [self.batch_size])
        supervised_sequence_mask = tf.sequence_mask(self.supervised_sequence_valid_length,
                                                    2 * self.max_length, dtype=tf.float32)
        self.supervised_loss = -((tf.to_float(self.supervised_sequence) *
                                  tf.log(tf.clip_by_value(probability, 1e-10, 1.0))
                                  + (1 - tf.to_float(self.supervised_sequence)) *
                                  tf.log(tf.clip_by_value(1 - probability, 1e-10, 1.0))))
        self.supervised_loss *= supervised_sequence_mask
        self.supervised_loss = tf.reduce_sum(self.supervised_loss) / self.batch_size
        supervised_opt = tf.train.AdamOptimizer(self.learning_rate)
        supervised_grads_and_vars = supervised_opt.compute_gradients(self.supervised_loss)
        supervised_grads_and_vars = [(tf.clip_by_value(gv[0], -1.0, 1.0) if gv[0] is not None else gv[0], gv[1])
                                     for gv in supervised_grads_and_vars]
        self.supervised_train_op = supervised_opt.apply_gradients(supervised_grads_and_vars)

        self.init_op = tf.global_variables_initializer()

    def _pre_train(self, X, sequence_length):
        """ pre-train a generator according to X, using random merged sequences
        :param X: the list of data matrix(first element is malware, second element is benign)
                  the data matrix shape: [num examples, max sequence length].
        :param sequence_length: the list of length (first element is malware, second element is benign)
                  the length of each row of X. shape: [num examples]
        """
        # get the benign data
        X_benign = X[1]
        X = X[0]
        benign_sequence_length = sequence_length[1]
        sequence_length = sequence_length[0]

        # shuffle and split train and val data
        index = np.arange(len(X))
        np.random.shuffle(index)
        X = X[index]
        sequence_length = sequence_length[index]
        num_training_samples = int(len(X) * 0.75)
        X_val = X[num_training_samples:]
        sequence_length_val = sequence_length[num_training_samples:]
        X = X[:num_training_samples]
        sequence_length = sequence_length[:num_training_samples]

        # get the supervised data
        supervised_index = []  # the list of tuple of (malware index, benign index)
        supervised_sequence = []  # the supervised sequence
        supervised_valid_length = []
        to_merge_index = range(num_training_samples)  # the index of malware to be merged
        log_message = ''
        for t in range(100):
            generated_malware = np.zeros([len(to_merge_index), 2 * self.max_length], dtype=np.int32)
            generated_length = np.zeros([len(to_merge_index)], dtype=np.int32)
            generated_sequence = np.zeros([len(to_merge_index), 2 * self.max_length], dtype=np.int32)
            generated_valid_length = np.zeros([len(to_merge_index)], dtype=np.int32)
            selected_benign = []
            for i in range(len(to_merge_index)):
                benign_index = random.randint(0, len(X_benign) - 1)
                selected_benign.append(benign_index)
                generated_length[i] = sequence_length[to_merge_index[i]] + benign_sequence_length[benign_index]
                idx_0 = 0
                idx_1 = 0
                while idx_0 < benign_sequence_length[benign_index] or idx_1 < sequence_length[to_merge_index[i]]:
                    selected = random.randint(0, 1)
                    if idx_1 >= sequence_length[to_merge_index[i]] \
                            or (idx_0 < benign_sequence_length[benign_index] and selected == 0):
                        generated_malware[i, idx_0 + idx_1] = X_benign[benign_index, idx_0]
                        generated_sequence[i, idx_0 + idx_1] = 0
                        idx_0 += 1
                    else:
                        generated_malware[i, idx_0 + idx_1] = X[to_merge_index[i], idx_1]
                        generated_sequence[i, idx_0 + idx_1] = 1
                        idx_1 += 1
                    if (idx_0 == benign_sequence_length[benign_index] or idx_1 == sequence_length[to_merge_index[i]])\
                            and generated_valid_length[i] == 0:
                        generated_valid_length[i] = idx_0 + idx_1
            generated_result = self.D.predict(generated_malware, generated_length)
            log_message += 'Generating supervised data. Epoch %d, TPR %g\n' % (t, generated_result.mean())
            print('Generating supervised data. Epoch %d, TPR %g' % (t, generated_result.mean()))
            to_merge_index_new = []
            for i in range(len(generated_result)):
                if generated_result[i] == 0:
                    supervised_index.append((to_merge_index[i], selected_benign[i]))
                    supervised_sequence.append(generated_sequence[i])
                    supervised_valid_length.append(generated_valid_length[i])
                else:
                    to_merge_index_new.append(to_merge_index[i])
            to_merge_index = to_merge_index_new
            if len(to_merge_index) == 0:
                break

        log_message += 'Final TPR %g\n' % (1 - float(len(supervised_index)) / len(X),)
        print('Final TPR %g' % (1 - float(len(supervised_index)) / len(X),))
        with open(self.model_path + '_pre_train' + '--training_log.txt', 'a') as f:
            f.write(log_message + '\n')
        # pack and shuffle
        index = np.arange(len(supervised_index))
        np.random.shuffle(index)
        supervised_index = np.array(supervised_index)[index]
        supervised_sequence = np.array(supervised_sequence)[index]
        supervised_valid_length = np.array(supervised_valid_length)[index]

        best_val_tpr = 1.0
        best_val_epoch = 0
        for epoch in range(self.max_epoch):
            train_loss = 0.0
            train_tpr = 0.0
            train_per_step_tpr = 0.0
            train_benign_priority = 0.0
            for start, end in zip(range(0, len(supervised_index), self.batch_size),
                                  range(self.batch_size, len(supervised_index) + 1, self.batch_size)):
                X_batch = X[supervised_index[start: end, 0]]
                sequence_length_batch = sequence_length[supervised_index[start: end, 0]]
                benign_index = supervised_index[start: end, 1]
                X_benign_batch = X_benign[benign_index]
                benign_sequence_length_batch = benign_sequence_length[benign_index]
                noise_inputs_batch = np.random.rand(self.batch_size, 2 * self.max_length,
                                                    self.noise_dim if self.noise_dim > 0 else 1)
                supervised_sequence_batch = supervised_sequence[start: end]
                supervised_valid_length_batch = supervised_valid_length[start: end]
                # supervised learning
                _, loss_value = self.sess.run([self.supervised_train_op, self.supervised_loss], feed_dict={
                    self.inputs_0: X_benign_batch,
                    self.inputs_0_length: benign_sequence_length_batch,
                    self.inputs_1: X_batch,
                    self.inputs_1_length: sequence_length_batch,
                    self.noise_inputs: noise_inputs_batch,
                    self.supervised_sequence: supervised_sequence_batch,
                    self.supervised_sequence_valid_length: supervised_valid_length_batch
                })

                # merge the two sequence
                predicted_sequence_batch = self.sess.run(self.predicted_sequence, feed_dict={
                    self.inputs_0: X_benign_batch,
                    self.inputs_0_length: benign_sequence_length_batch,
                    self.inputs_1: X_batch,
                    self.inputs_1_length: sequence_length_batch,
                    self.noise_inputs: noise_inputs_batch
                })
                # get the merged sequence
                predicted_sequence_valid_length_batch = np.zeros((self.batch_size,), dtype=np.int32)
                merged_batch = np.zeros((self.batch_size, 2 * self.max_length), dtype=np.int32)
                merged_sequence_length_batch = np.zeros((self.batch_size,), dtype=np.int32)
                for i in range(self.batch_size):
                    merged_sequence_length_batch[i] = benign_sequence_length_batch[i] + sequence_length_batch[i]
                    idx_0 = 0
                    idx_1 = 0
                    benign_priority = 0
                    for j in range(2 * self.max_length):
                        if predicted_sequence_batch[i, j] == 0:
                            merged_batch[i, j] = X_benign_batch[i, idx_0]
                            benign_priority += sequence_length_batch[i] - idx_1
                            idx_0 += 1
                            if idx_0 >= benign_sequence_length_batch[i]:
                                predicted_sequence_valid_length_batch[i] = j + 1
                                remaining_length = sequence_length_batch[i] - idx_1
                                merged_batch[i, j + 1: j + 1 + remaining_length] = \
                                    X_batch[i, idx_1: idx_1 + remaining_length]
                                break
                        else:
                            merged_batch[i, j] = X_batch[i, idx_1]
                            idx_1 += 1
                            if idx_1 >= sequence_length_batch[i]:
                                predicted_sequence_valid_length_batch[i] = j + 1
                                remaining_length = benign_sequence_length_batch[i] - idx_0
                                merged_batch[i, j + 1: j + 1 + remaining_length] = \
                                    X_benign_batch[i, idx_0: idx_0 + remaining_length]
                                break
                    train_benign_priority += float(benign_priority) \
                                             / benign_sequence_length_batch[i] / sequence_length_batch[i]

                discriminator_result_batch = self.D.predict(merged_batch, merged_sequence_length_batch)
                discriminator_per_step_result_batch = self.D.predict_per_step(
                    merged_batch, merged_sequence_length_batch)

                train_loss += loss_value
                train_tpr += discriminator_result_batch.mean()
                train_per_step_tpr += discriminator_per_step_result_batch.mean()
            train_loss /= len(supervised_index) / self.batch_size
            train_tpr /= len(supervised_index) / self.batch_size
            train_per_step_tpr /= len(supervised_index) / self.batch_size
            train_benign_priority /= len(supervised_index)
            self.saver.save(self.sess, self.model_path + '_pre_train', global_step=epoch)

            val_loss = 0.0
            val_value_loss = 0.0
            val_tpr = 0.0
            val_per_step_tpr = 0.0
            val_benign_priority = 0.0
            for start, end in zip(range(0, len(X_val), self.batch_size),
                                  range(self.batch_size, len(X_val) + 1, self.batch_size)):
                X_batch = X_val[start: end]
                sequence_length_batch = sequence_length_val[start: end]
                benign_index = np.random.random_integers(0, len(X_benign) - 1, self.batch_size)
                X_benign_batch = X_benign[benign_index]
                benign_sequence_length_batch = benign_sequence_length[benign_index]
                noise_inputs_batch = np.random.rand(self.batch_size, 2 * self.max_length,
                                                    self.noise_dim if self.noise_dim > 0 else 1)

                # merge the two sequence
                predicted_sequence_batch = self.sess.run(self.predicted_sequence, feed_dict={
                    self.inputs_0: X_benign_batch,
                    self.inputs_0_length: benign_sequence_length_batch,
                    self.inputs_1: X_batch,
                    self.inputs_1_length: sequence_length_batch,
                    self.noise_inputs: noise_inputs_batch
                })
                # get the merged sequence
                predicted_sequence_valid_length_batch = np.zeros((self.batch_size,), dtype=np.int32)
                merged_batch = np.zeros((self.batch_size, 2 * self.max_length), dtype=np.int32)
                merged_sequence_length_batch = np.zeros((self.batch_size,), dtype=np.int32)
                for i in range(self.batch_size):
                    merged_sequence_length_batch[i] = benign_sequence_length_batch[i] + sequence_length_batch[i]
                    idx_0 = 0
                    idx_1 = 0
                    benign_priority = 0
                    for j in range(2 * self.max_length):
                        if predicted_sequence_batch[i, j] == 0:
                            merged_batch[i, j] = X_benign_batch[i, idx_0]
                            benign_priority += sequence_length_batch[i] - idx_1
                            idx_0 += 1
                            if idx_0 >= benign_sequence_length_batch[i]:
                                predicted_sequence_valid_length_batch[i] = j + 1
                                remaining_length = sequence_length_batch[i] - idx_1
                                merged_batch[i, j + 1: j + 1 + remaining_length] = \
                                    X_batch[i, idx_1: idx_1 + remaining_length]
                                break
                        else:
                            merged_batch[i, j] = X_batch[i, idx_1]
                            idx_1 += 1
                            if idx_1 >= sequence_length_batch[i]:
                                predicted_sequence_valid_length_batch[i] = j + 1
                                remaining_length = benign_sequence_length_batch[i] - idx_0
                                merged_batch[i, j + 1: j + 1 + remaining_length] = \
                                    X_benign_batch[i, idx_0: idx_0 + remaining_length]
                                break
                    val_benign_priority += float(benign_priority) \
                                           / benign_sequence_length_batch[i] / sequence_length_batch[i]

                discriminator_result_batch = self.D.predict(merged_batch, merged_sequence_length_batch)
                discriminator_per_step_result_batch = self.D.predict_per_step(
                    merged_batch, merged_sequence_length_batch)

                loss_value, value_loss_value = self.sess.run([self.loss, self.value_loss], feed_dict={
                    self.inputs_0: X_benign_batch,
                    self.inputs_0_length: benign_sequence_length_batch,
                    self.inputs_1: X_batch,
                    self.inputs_1_length: sequence_length_batch,
                    self.noise_inputs: noise_inputs_batch,
                    self.predicted_sequence_valid_length: predicted_sequence_valid_length_batch,
                    self.discriminator_per_step_result: discriminator_per_step_result_batch}
                )
                val_loss += loss_value
                val_value_loss += value_loss_value
                val_tpr += discriminator_result_batch.mean()
                val_per_step_tpr += discriminator_per_step_result_batch.mean()
            val_loss /= len(X_val) / self.batch_size
            val_value_loss /= len(X_val) / self.batch_size
            val_tpr /= len(X_val) / self.batch_size
            val_per_step_tpr /= len(X_val) / self.batch_size
            val_benign_priority /= len(X_val)

            if val_tpr < best_val_tpr:
                best_val_tpr = val_tpr
                best_val_epoch = epoch

            log_message = 'Generator Epoch %d: Loss & PRI / TPR: Tra=%g, %g / %g, %g, Val=%g, %g, %g / %g, %g' % \
                          (epoch, train_loss, train_benign_priority, train_tpr, train_per_step_tpr,
                           val_loss, val_value_loss, val_benign_priority, val_tpr, val_per_step_tpr)
            print(log_message)
            with open(self.model_path + '_pre_train' + '--training_log.txt', 'a') as f:
                f.write(log_message + '\n')

            if epoch - best_val_epoch >= self.max_epoch_no_improvement:
                self.saver.restore(self.sess, self.model_path + '_pre_train' + '-' + str(best_val_epoch))
                break

    def train(self, X, sequence_length):
        """ train a generator according to X
        :param X: the list of data matrix(first element is malware, second element is benign)
                  the data matrix shape: [num examples, max sequence length].
        :param sequence_length: the list of length (first element is malware, second element is benign)
                  the length of each row of X. shape: [num examples]
        """
        self.sess.run(self.init_op)
        self._pre_train(X, sequence_length)
        # get the benign data
        X_benign = X[1]
        X = X[0]
        benign_sequence_length = sequence_length[1]
        sequence_length = sequence_length[0]

        # shuffle and split train and val data
        index = np.arange(len(X))
        np.random.shuffle(index)
        X = X[index]
        sequence_length = sequence_length[index]
        num_training_samples = int(len(X) * 0.75)
        X_val = X[num_training_samples:]
        sequence_length_val = sequence_length[num_training_samples:]
        X = X[:num_training_samples]
        sequence_length = sequence_length[:num_training_samples]

        best_val_tpr = 1.0
        best_val_epoch = 0
        for epoch in range(self.max_epoch):
            train_loss = 0.0
            train_value_loss = 0.0
            train_tpr = 0.0
            train_per_step_tpr = 0.0
            train_benign_priority = 0.0
            for start, end in zip(range(0, len(X), self.batch_size),
                                  range(self.batch_size, len(X) + 1, self.batch_size)):
                X_batch = X[start: end]
                sequence_length_batch = sequence_length[start: end]
                benign_index = np.random.random_integers(0, len(X_benign) - 1, self.batch_size)
                X_benign_batch = X_benign[benign_index]
                benign_sequence_length_batch = benign_sequence_length[benign_index]
                noise_inputs_batch = np.random.rand(self.batch_size, 2 * self.max_length,
                                                    self.noise_dim if self.noise_dim > 0 else 1)
                # merge the two sequence
                predicted_sequence_batch = self.sess.run(self.predicted_sequence, feed_dict={
                    self.inputs_0: X_benign_batch,
                    self.inputs_0_length: benign_sequence_length_batch,
                    self.inputs_1: X_batch,
                    self.inputs_1_length: sequence_length_batch,
                    self.noise_inputs: noise_inputs_batch
                })
                # get the merged sequence
                predicted_sequence_valid_length_batch = np.zeros((self.batch_size,), dtype=np.int32)
                merged_batch = np.zeros((self.batch_size, 2 * self.max_length), dtype=np.int32)
                merged_sequence_length_batch = np.zeros((self.batch_size,), dtype=np.int32)
                for i in range(self.batch_size):
                    merged_sequence_length_batch[i] = benign_sequence_length_batch[i] + sequence_length_batch[i]
                    idx_0 = 0
                    idx_1 = 0
                    benign_priority = 0
                    for j in range(2 * self.max_length):
                        if predicted_sequence_batch[i, j] == 0:
                            merged_batch[i, j] = X_benign_batch[i, idx_0]
                            benign_priority += sequence_length_batch[i] - idx_1
                            idx_0 += 1
                            if idx_0 >= benign_sequence_length_batch[i]:
                                predicted_sequence_valid_length_batch[i] = j + 1
                                remaining_length = sequence_length_batch[i] - idx_1
                                merged_batch[i, j + 1: j + 1 + remaining_length] = \
                                    X_batch[i, idx_1: idx_1 + remaining_length]
                                break
                        else:
                            merged_batch[i, j] = X_batch[i, idx_1]
                            idx_1 += 1
                            if idx_1 >= sequence_length_batch[i]:
                                predicted_sequence_valid_length_batch[i] = j + 1
                                remaining_length = benign_sequence_length_batch[i] - idx_0
                                merged_batch[i, j + 1: j + 1 + remaining_length] = \
                                    X_benign_batch[i, idx_0: idx_0 + remaining_length]
                                break
                    train_benign_priority += float(benign_priority) \
                                             / benign_sequence_length_batch[i] / sequence_length_batch[i]

                discriminator_result_batch = self.D.predict(merged_batch, merged_sequence_length_batch)
                discriminator_per_step_result_batch = self.D.predict_per_step(
                    merged_batch, merged_sequence_length_batch)

                _, _, loss_value, value_loss_value = self.sess.run(
                    [self.train_op, self.value_train_op, self.loss, self.value_loss], feed_dict={
                        self.inputs_0: X_benign_batch,
                        self.inputs_0_length: benign_sequence_length_batch,
                        self.inputs_1: X_batch,
                        self.inputs_1_length: sequence_length_batch,
                        self.noise_inputs: noise_inputs_batch,
                        self.predicted_sequence_valid_length: predicted_sequence_valid_length_batch,
                        self.discriminator_per_step_result: discriminator_per_step_result_batch}
                )
                train_loss += loss_value
                train_value_loss += value_loss_value
                train_tpr += discriminator_result_batch.mean()
                train_per_step_tpr += discriminator_per_step_result_batch.mean()
            train_loss /= len(X) / self.batch_size
            train_value_loss /= len(X) / self.batch_size
            train_tpr /= len(X) / self.batch_size
            train_per_step_tpr /= len(X) / self.batch_size
            train_benign_priority /= len(X)
            self.saver.save(self.sess, self.model_path, global_step=epoch)

            val_loss = 0.0
            val_value_loss = 0.0
            val_tpr = 0.0
            val_per_step_tpr = 0.0
            val_benign_priority = 0.0
            for start, end in zip(range(0, len(X_val), self.batch_size),
                                  range(self.batch_size, len(X_val) + 1, self.batch_size)):
                X_batch = X_val[start: end]
                sequence_length_batch = sequence_length_val[start: end]
                benign_index = np.random.random_integers(0, len(X_benign) - 1, self.batch_size)
                X_benign_batch = X_benign[benign_index]
                benign_sequence_length_batch = benign_sequence_length[benign_index]
                noise_inputs_batch = np.random.rand(self.batch_size, 2 * self.max_length,
                                                    self.noise_dim if self.noise_dim > 0 else 1)
                # merge the two sequence
                predicted_sequence_batch = self.sess.run(self.predicted_sequence, feed_dict={
                    self.inputs_0: X_benign_batch,
                    self.inputs_0_length: benign_sequence_length_batch,
                    self.inputs_1: X_batch,
                    self.inputs_1_length: sequence_length_batch,
                    self.noise_inputs: noise_inputs_batch
                })
                # get the merged sequence
                predicted_sequence_valid_length_batch = np.zeros((self.batch_size,), dtype=np.int32)
                merged_batch = np.zeros((self.batch_size, 2 * self.max_length), dtype=np.int32)
                merged_sequence_length_batch = np.zeros((self.batch_size,), dtype=np.int32)
                for i in range(self.batch_size):
                    merged_sequence_length_batch[i] = benign_sequence_length_batch[i] + sequence_length_batch[i]
                    idx_0 = 0
                    idx_1 = 0
                    benign_priority = 0
                    for j in range(2 * self.max_length):
                        if predicted_sequence_batch[i, j] == 0:
                            merged_batch[i, j] = X_benign_batch[i, idx_0]
                            benign_priority += sequence_length_batch[i] - idx_1
                            idx_0 += 1
                            if idx_0 >= benign_sequence_length_batch[i]:
                                predicted_sequence_valid_length_batch[i] = j + 1
                                remaining_length = sequence_length_batch[i] - idx_1
                                merged_batch[i, j + 1: j + 1 + remaining_length] = \
                                    X_batch[i, idx_1: idx_1 + remaining_length]
                                break
                        else:
                            merged_batch[i, j] = X_batch[i, idx_1]
                            idx_1 += 1
                            if idx_1 >= sequence_length_batch[i]:
                                predicted_sequence_valid_length_batch[i] = j + 1
                                remaining_length = benign_sequence_length_batch[i] - idx_0
                                merged_batch[i, j + 1: j + 1 + remaining_length] = \
                                    X_benign_batch[i, idx_0: idx_0 + remaining_length]
                                break
                    val_benign_priority += float(benign_priority) \
                                           / benign_sequence_length_batch[i] / sequence_length_batch[i]

                discriminator_result_batch = self.D.predict(merged_batch, merged_sequence_length_batch)
                discriminator_per_step_result_batch = self.D.predict_per_step(
                    merged_batch, merged_sequence_length_batch)

                loss_value, value_loss_value = self.sess.run([self.loss, self.value_loss], feed_dict={
                    self.inputs_0: X_benign_batch,
                    self.inputs_0_length: benign_sequence_length_batch,
                    self.inputs_1: X_batch,
                    self.inputs_1_length: sequence_length_batch,
                    self.noise_inputs: noise_inputs_batch,
                    self.predicted_sequence_valid_length: predicted_sequence_valid_length_batch,
                    self.discriminator_per_step_result: discriminator_per_step_result_batch}
                )
                val_loss += loss_value
                val_value_loss += value_loss_value
                val_tpr += discriminator_result_batch.mean()
                val_per_step_tpr += discriminator_per_step_result_batch.mean()
            val_loss /= len(X_val) / self.batch_size
            val_value_loss /= len(X_val) / self.batch_size
            val_tpr /= len(X_val) / self.batch_size
            val_per_step_tpr /= len(X_val) / self.batch_size
            val_benign_priority /= len(X_val)

            if val_tpr < best_val_tpr:
                best_val_tpr = val_tpr
                best_val_epoch = epoch

            log_message = 'Generator Epoch %d: Loss & PRI / TPR: Tra=%g, %g, %g / %g, %g, Val=%g, %g, %g / %g, %g' % \
                          (epoch, train_loss, train_value_loss, train_benign_priority, train_tpr, train_per_step_tpr,
                           val_loss, val_value_loss, val_benign_priority, val_tpr, val_per_step_tpr)
            print(log_message)
            with open(self.model_path + '--training_log.txt', 'a') as f:
                f.write(log_message + '\n')

            if epoch - best_val_epoch >= self.max_epoch_no_improvement:
                self.saver.restore(self.sess, self.model_path + '-' + str(best_val_epoch))
                break

    def sample(self, X, sequence_length):
        """ generate samples for X
        :param X: the list of data matrix(first element is malware, second element is benign)
                  the data matrix shape: [num examples, max sequence length].
        :param sequence_length: the list of length (first element is malware, second element is benign)
                  the length of each row of X. shape: [num examples]
        :return: the generated samples, the corresponding sequence_length
        """
        # get the benign data
        X_benign = X[1]
        X = X[0]
        benign_sequence_length = sequence_length[1]
        sequence_length = sequence_length[0]

        num_samples = len(X)
        X = np.concatenate((X, X[:self.batch_size - 1]))
        sequence_length = np.concatenate((sequence_length, sequence_length[:self.batch_size - 1]))
        generated_X = np.zeros((len(X), 2 * self.max_length), dtype=np.int32)
        generated_sequence_length = np.zeros((len(X),), dtype=np.int32)

        for start, end in zip(range(0, len(X), self.batch_size),
                                range(self.batch_size, len(X) + 1, self.batch_size)):
            X_batch = X[start: end]
            sequence_length_batch = sequence_length[start: end]
            benign_index = np.random.random_integers(0, len(X_benign) - 1, self.batch_size)
            X_benign_batch = X_benign[benign_index]
            benign_sequence_length_batch = benign_sequence_length[benign_index]
            noise_inputs_batch = np.random.rand(self.batch_size, 2 * self.max_length,
                                                self.noise_dim if self.noise_dim > 0 else 1)
            # merge the two sequence
            predicted_sequence_batch = self.sess.run(self.predicted_sequence, feed_dict={
                self.inputs_0: X_benign_batch,
                self.inputs_0_length: benign_sequence_length_batch,
                self.inputs_1: X_batch,
                self.inputs_1_length: sequence_length_batch,
                self.noise_inputs: noise_inputs_batch
            })
            # get the merged sequence
            merged_batch = np.zeros((self.batch_size, 2 * self.max_length), dtype=np.int32)
            merged_sequence_length_batch = np.zeros((self.batch_size,), dtype=np.int32)
            for i in range(self.batch_size):
                merged_sequence_length_batch[i] = benign_sequence_length_batch[i] + sequence_length_batch[i]
                idx_0 = 0
                idx_1 = 0
                for j in range(2 * self.max_length):
                    if predicted_sequence_batch[i, j] == 0:
                        merged_batch[i, j] = X_benign_batch[i, idx_0]
                        idx_0 += 1
                        if idx_0 >= benign_sequence_length_batch[i]:
                            remaining_length = sequence_length_batch[i] - idx_1
                            merged_batch[i, j + 1: j + 1 + remaining_length] = \
                                X_batch[i, idx_1: idx_1 + remaining_length]
                            break
                    else:
                        merged_batch[i, j] = X_batch[i, idx_1]
                        idx_1 += 1
                        if idx_1 >= sequence_length_batch[i]:
                            remaining_length = benign_sequence_length_batch[i] - idx_0
                            merged_batch[i, j + 1: j + 1 + remaining_length] = \
                                X_benign_batch[i, idx_0: idx_0 + remaining_length]
                            break
            generated_X[start: end] = merged_batch
            generated_sequence_length[start: end] = merged_sequence_length_batch

        return generated_X[:num_samples], generated_sequence_length[:num_samples]


class SeqInserter(SeqGenerator):
    """
    Merge two sequences
    """
    def __init__(self, D, cell_type='LSTM', G_layers=[512], G_bidirectional=True, D_layers=[512],
                 D_attention_layers=[512], D_ff_layers = [512], batch_size=128, benign_batch_size=128,
                 num_tokens=161, max_length=2048, max_epoch=1000, max_epoch_no_improvement=25,
                 learning_rate=0.001, temperature=1.0, regularization=0.0, model_path='model'):
        """
        :param cell_type: 'LSTM', 'RNN', 'GRU'
        :param G_layers: a list of integer
        :param G_bidirectional: whether to use bidirectional RNN for G
        :param D_layers: a list of integer
        :param D_attention_layers: a list of integer
        :param D_ff_layers: a list of integer
        :param batch_size: the size of mini-batch
        :param benign_batch_size: the size of benign mini-batch
        :param num_tokens: the number of distinct tokens of input data
        :param max_length: the maximum length of the input sequence
        :param max_epoch: the maximum epoch for training RNN
        :param max_epoch_no_improvement: if the performance of the val set doesn't improve for
                max_epoch_no_improvement, stop training
        :param learning_rate: the learning rate for RNN
        :param temperature: the temperature for Gumbel-softmax
        :param regularization: the regularization coefficient to constrain the number of generated non-null elements
        :param model_path: the path to save the model
        """
        self.D = D
        self.cell_type = cell_type
        self.G_layers = G_layers
        self.G_bidirectional = G_bidirectional
        self.D_layers = D_layers
        self.D_attention_layers = D_attention_layers + [1]
        self.D_ff_layers = D_ff_layers + [1]
        self.batch_size = batch_size
        self.benign_batch_size = benign_batch_size
        self.num_tokens = num_tokens
        self.max_length = max_length
        self.max_epoch = max_epoch
        self.max_epoch_no_improvement = max_epoch_no_improvement
        self.learning_rate = learning_rate
        self.temperature = temperature
        self.regularization = regularization
        self.model_path = model_path
        g = tf.Graph()
        with g.as_default():
            self._build_model()
            self.saver = tf.train.Saver(max_to_keep=0)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config, graph=g)

    def _build_model(self):
        # initializing the cell type
        if self.cell_type is 'RNN':
            cell_element = tf.contrib.rnn.BasicRNNCell
        elif self.cell_type is 'LSTM':
            cell_element = tf.contrib.rnn.BasicLSTMCell
        elif self.cell_type is 'GRU':
            cell_element = tf.contrib.rnn.GRUCell
        else:
            raise ValueError('cell_type must be one of "LSTM", "RNN", "GRU"')

        # inputs
        self.inputs = tf.placeholder(tf.int32, [self.batch_size, self.max_length])
        self.inputs_length = tf.placeholder(tf.int32, [self.batch_size])
        self.Gumbel_samples = tf.placeholder(tf.float32, [self.batch_size, self.max_length, self.num_tokens + 1])
        inputs_one_hot = tf.one_hot(self.inputs, self.num_tokens)

        # G
        with tf.variable_scope("G"):
            # set the depth of G cell
            if len(self.G_layers) == 1:
                G_cell = cell_element(self.G_layers[0])
            elif len(self.G_layers) > 1:
                cell_elements = []
                for G_layer in self.G_layers:
                    cell_elements.append(cell_element(G_layer))
                G_cell = tf.contrib.rnn.MultiRNNCell(cell_elements)
            else:
                raise ValueError('G must have at least one layer')
            if self.G_bidirectional:
                G_hiddens, _ = tf.nn.bidirectional_dynamic_rnn(G_cell, G_cell, inputs_one_hot, self.inputs_length,
                                                               dtype=tf.float32, swap_memory=True, time_major=False)
                G_hiddens = tf.concat(G_hiddens, 2)
            else:
                G_hiddens, _ = tf.nn.dynamic_rnn(G_cell, inputs_one_hot, self.inputs_length, dtype=tf.float32,
                                                 swap_memory=True, time_major=False)
            dim_hiddens = self.G_layers[-1] * (2 if self.G_bidirectional else 1)
            G_hiddens = tf.reshape(G_hiddens, [self.batch_size * self.max_length, dim_hiddens])
            W_G_output = tf.Variable(tf.random_uniform([dim_hiddens, self.num_tokens + 1], -0.1, 0.1),
                                     name='W_G_output')
            b_G_output = tf.Variable(tf.zeros([self.num_tokens + 1]), name='b_G_output')
            G_log_probability = tf.reshape(tf.nn.log_softmax(tf.matmul(G_hiddens, W_G_output) + b_G_output),
                                           [self.batch_size, self.max_length, self.num_tokens + 1])
            Gumbel_logits = (G_log_probability + self.Gumbel_samples) / self.temperature
            self.G_outputs = tf.nn.softmax(Gumbel_logits)

        # calculate the regularization
        inputs_mask = tf.sequence_mask(self.inputs_length, self.max_length, dtype=tf.float32)
        self.G_regu = tf.reduce_sum((1.0 - self.G_outputs[:, :, -1]) * inputs_mask) / self.batch_size

        # get the prediction result of G
        self.G_discrete_outputs = tf.concat((tf.expand_dims(self.inputs, 2),
                                                tf.expand_dims(tf.to_int32(tf.argmax(Gumbel_logits, axis=2)), 2)), 2)
        self.G_discrete_outputs = tf.reshape(self.G_discrete_outputs, [self.batch_size, 2 * self.max_length])
        self.G_outputs = tf.reshape(tf.concat((tf.one_hot(self.inputs, self.num_tokens + 1), self.G_outputs), 2),
                                    [self.batch_size, 2 * self.max_length, self.num_tokens + 1])

        # D

        def _get_D_graph(D_inputs, D_inputs_length, D_batch_size, reuse):
            with tf.variable_scope('D', reuse=reuse):
                # set the depth of D cell
                if len(self.D_layers) == 1:
                    D_cell = cell_element(self.D_layers[0])
                elif len(self.D_layers) > 1:
                    cell_elements = []
                    for D_layer in self.D_layers:
                        cell_elements.append(cell_element(D_layer))
                    D_cell = tf.contrib.rnn.MultiRNNCell(cell_elements)
                else:
                    raise ValueError('D must have at least one layer')

                # recurrent connect
                D_hiddens, _ = tf.nn.bidirectional_dynamic_rnn(D_cell, D_cell, D_inputs, D_inputs_length,
                                                               dtype=tf.float32, swap_memory=True, time_major=False)
                D_hiddens = tf.concat(D_hiddens, 2)

                # attention
                last_attention_hidden = self.D_layers[-1] * 2
                attention_hidden = tf.reshape(D_hiddens, [D_batch_size * 2 * self.max_length, last_attention_hidden])
                for layer, attention_layer in enumerate(self.D_attention_layers):
                    W_D_attention = tf.get_variable('W_D_attention_%d' % (layer,), initializer=None if reuse else
                    tf.random_uniform([last_attention_hidden, attention_layer], -0.1, 0.1))
                    last_attention_hidden = attention_layer
                    b_D_attention = tf.get_variable('b_D_attention_%d' % (layer,), initializer=None if reuse else
                    tf.zeros([attention_layer]))
                    attention_hidden = tf.matmul(attention_hidden, W_D_attention) + b_D_attention
                    if layer < len(self.D_attention_layers) - 1:
                        attention_hidden = tf.nn.tanh(attention_hidden)
                attention_weights = tf.exp(tf.reshape(attention_hidden, [D_batch_size, 2 * self.max_length]))
                attention_mask = tf.sequence_mask(D_inputs_length, 2 * self.max_length, dtype=tf.float32)
                attention_weights *= attention_mask
                attention_weights_sum = tf.reduce_sum(attention_weights, 1, keep_dims=True)
                attention_weights /= attention_weights_sum
                ff_input = tf.reduce_sum(tf.multiply(D_hiddens, tf.expand_dims(attention_weights, 2)), 1)

                # ff
                last_ff_hidden = self.D_layers[-1] * 2
                ff_hidden = ff_input
                for layer, ff_layer in enumerate(self.D_ff_layers):
                    W_D_ff = tf.get_variable('W_D_ff_%d' % (layer,), initializer=None if reuse else
                    tf.random_uniform([last_ff_hidden, ff_layer], -0.1, 0.1))
                    last_ff_hidden = ff_layer
                    b_D_ff = tf.get_variable('b_D_ff_%d' % (layer,), initializer=None if reuse else
                    tf.zeros([ff_layer]))
                    ff_hidden = tf.matmul(ff_hidden, W_D_ff) + b_D_ff
                    if layer < len(self.D_ff_layers) - 1:
                        ff_hidden = tf.nn.tanh(ff_hidden)
                    else:
                        ff_hidden = tf.nn.sigmoid(ff_hidden)
            return tf.reshape(ff_hidden, [D_batch_size])

        # D loss
        D_all_batch_size = self.batch_size + self.benign_batch_size
        self.discriminator_result = tf.placeholder(tf.int32, [D_all_batch_size])
        self.D_all_inputs = tf.placeholder(tf.float32, [D_all_batch_size, 2 * self.max_length, self.num_tokens + 1])
        self.D_all_inputs_length = tf.placeholder(tf.int32, [D_all_batch_size])
        D_all_probability = _get_D_graph(self.D_all_inputs, self.D_all_inputs_length, D_all_batch_size, False)
        self.D_loss = -((tf.to_float(self.discriminator_result) *
                         tf.log(tf.clip_by_value(D_all_probability, 1e-10, 1.0))
                         + (1 - tf.to_float(self.discriminator_result)) *
                         tf.log(tf.clip_by_value(1 - D_all_probability, 1e-10, 1.0))))
        self.D_loss = tf.reduce_mean(self.D_loss)
        D_opt = tf.train.AdamOptimizer(self.learning_rate)
        D_grads_and_vars = D_opt.compute_gradients(self.D_loss,
                                                   tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='D'))
        D_grads_and_vars = [(tf.clip_by_value(gv[0], -1.0, 1.0), gv[1]) for gv in D_grads_and_vars]
        self.D_train_op = D_opt.apply_gradients(D_grads_and_vars)

        # G loss
        G_D_probability = _get_D_graph(self.G_outputs, 2 * self.inputs_length, self.batch_size, True)
        self.G_loss = -tf.log(tf.clip_by_value(G_D_probability, 1e-10, 1.0))
        self.G_loss = tf.reduce_mean(self.G_loss)
        G_opt = tf.train.AdamOptimizer(self.learning_rate)
        G_grads_and_vars = G_opt.compute_gradients(-self.G_loss + self.regularization * self.G_regu,
                                                   tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='G'))
        G_grads_and_vars = [(tf.clip_by_value(gv[0], -1.0, 1.0), gv[1]) for gv in G_grads_and_vars]
        self.G_train_op = G_opt.apply_gradients(G_grads_and_vars)

        self.init_op = tf.global_variables_initializer()

    def train(self, X, sequence_length):
        """ train a generator according to X
        :param X: the list of data matrix(first element is malware, second element is benign)
                  the data matrix shape: [num examples, max sequence length].
        :param sequence_length: the list of length (first element is malware, second element is benign)
                  the length of each row of X. shape: [num examples]
        """
        self.sess.run(self.init_op)
        self.saver.restore(self.sess, '../seq_models/20170222_rnn_biLSTM_LSTM_G_128_birnn_D_128_attention_128_ff_128_max_len_1024_lr_0.001_temp_60_regu_0.0001/model-1')
        return
        # get the benign data
        X_benign = X[1]
        X = X[0]
        benign_sequence_length = sequence_length[1]
        sequence_length = sequence_length[0]
        index = np.arange(len(X_benign))
        np.random.shuffle(index)
        X_benign = X_benign[index]
        benign_sequence_length = benign_sequence_length[index]
        num_training_samples = int(len(X_benign) * 0.75)
        X_benign_val = X_benign[num_training_samples:]
        benign_sequence_length_val = benign_sequence_length[num_training_samples:]
        X_benign = X_benign[:num_training_samples]
        benign_sequence_length = benign_sequence_length[:num_training_samples]

        # shuffle and split train and val data
        index = np.arange(len(X))
        np.random.shuffle(index)
        X = X[index]
        sequence_length = sequence_length[index]
        num_training_samples = int(len(X) * 0.75)
        X_val = X[num_training_samples:]
        sequence_length_val = sequence_length[num_training_samples:]
        X = X[:num_training_samples]
        sequence_length = sequence_length[:num_training_samples]
        
        


        best_val_tpr = 1.0
        best_val_epoch = 0
        for epoch in range(self.max_epoch):
            train_D_loss = 0.0
            train_G_loss = 0.0
            train_G_regu = 0.0
            train_tpr = 0.0
            train_increased_length = 0.0
            benign_pointer = 0
            for start, end in zip(range(0, len(X), self.batch_size),
                                  range(self.batch_size, len(X) + 1, self.batch_size)):
                X_batch = X[start: end]
                sequence_length_batch = sequence_length[start: end]
                Gumbel_samples_batch = 1e-10 + (1 - 2e-10) * \
                                               np.random.rand(self.batch_size, self.max_length, self.num_tokens + 1)
                Gumbel_samples_batch = -np.log(-np.log(Gumbel_samples_batch))
                if benign_pointer + self.benign_batch_size > len(X_benign):
                    benign_pointer = 0
                X_benign_batch = X_benign[benign_pointer: benign_pointer + self.benign_batch_size]
                benign_sequence_length_batch = benign_sequence_length[benign_pointer:
                                                                      benign_pointer + self.benign_batch_size]
                benign_pointer += self.benign_batch_size
                # get the generated malware
                G_discrete_outputs_batch, G_outputs_batch = self.sess.run([self.G_discrete_outputs, self.G_outputs],
                                                                          feed_dict={
                    self.inputs: X_batch,
                    self.inputs_length: sequence_length_batch,
                    self.Gumbel_samples: Gumbel_samples_batch
                })
                # remove non-output token
                G_discrete_length_batch = 2 * sequence_length_batch
                for i in range(self.batch_size):
                    j1 = 0
                    for j2 in range(G_discrete_length_batch[i]):
                        if G_discrete_outputs_batch[i, j2] < self.num_tokens:
                            if j1 != j2:
                                G_discrete_outputs_batch[i, j1] = G_discrete_outputs_batch[i, j2]
                            j1 += 1
                            if j2 % 2 == 1:
                                train_increased_length += 1
                    G_discrete_length_batch[i] = j1

                BB_inputs_batch = np.concatenate((G_discrete_outputs_batch,
                                                  np.hstack((X_benign_batch, X_benign_batch))))
                BB_inputs_length_batch = np.concatenate((G_discrete_length_batch, benign_sequence_length_batch))
                BB_discriminator_result_batch = self.D.predict(BB_inputs_batch, BB_inputs_length_batch)
                X_benign_one_hot_batch = np.eye(self.num_tokens + 1)[np.hstack((X_benign_batch, X_benign_batch))]
                D_inputs_batch = np.concatenate((G_outputs_batch, X_benign_one_hot_batch))
                D_inputs_length_batch = np.concatenate((2 * sequence_length_batch, benign_sequence_length_batch))

                # train D
                _, D_loss_value = self.sess.run([self.D_train_op, self.D_loss], feed_dict={
                    self.D_all_inputs: D_inputs_batch,
                    self.D_all_inputs_length: D_inputs_length_batch,
                    self.discriminator_result: BB_discriminator_result_batch
                })
                train_D_loss += D_loss_value

                # train G
                _, G_loss_value, G_regu_value = self.sess.run([self.G_train_op, self.G_loss, self.G_regu], feed_dict={
                    self.inputs: X_batch,
                    self.inputs_length: sequence_length_batch,
                    self.Gumbel_samples: Gumbel_samples_batch
                })
                train_G_loss += G_loss_value
                train_G_regu += G_regu_value
                train_tpr += BB_discriminator_result_batch[:self.batch_size].mean()
            train_D_loss /= len(X) / self.batch_size
            train_G_loss /= len(X) / self.batch_size
            train_G_regu /= len(X) / self.batch_size
            train_tpr /= len(X) / self.batch_size
            train_increased_length /= len(X)
            self.saver.save(self.sess, self.model_path, global_step=epoch)

            val_D_loss = 0.0
            val_G_loss = 0.0
            val_G_regu = 0.0
            val_tpr = 0.0
            val_increased_length = 0.0
            benign_pointer = 0
            for start, end in zip(range(0, len(X_val), self.batch_size),
                                  range(self.batch_size, len(X_val) + 1, self.batch_size)):
                X_batch = X_val[start: end]
                sequence_length_batch = sequence_length_val[start: end]
                Gumbel_samples_batch = 1e-10 + (1 - 2e-10) * \
                                               np.random.rand(self.batch_size, self.max_length, self.num_tokens + 1)
                Gumbel_samples_batch = -np.log(-np.log(Gumbel_samples_batch))
                if benign_pointer + self.benign_batch_size > len(X_benign_val):
                    benign_pointer = 0
                X_benign_batch = X_benign_val[benign_pointer: benign_pointer + self.benign_batch_size]
                benign_sequence_length_batch = benign_sequence_length_val[benign_pointer:
                benign_pointer + self.benign_batch_size]
                benign_pointer += self.benign_batch_size
                # get the generated malware
                G_discrete_outputs_batch, G_outputs_batch = self.sess.run([self.G_discrete_outputs, self.G_outputs],
                                                                          feed_dict={
                                                                              self.inputs: X_batch,
                                                                              self.inputs_length: sequence_length_batch,
                                                                              self.Gumbel_samples: Gumbel_samples_batch
                                                                          })
                # remove non-output token
                G_discrete_length_batch = 2 * sequence_length_batch
                for i in range(self.batch_size):
                    j1 = 0
                    for j2 in range(G_discrete_length_batch[i]):
                        if G_discrete_outputs_batch[i, j2] < self.num_tokens:
                            if j1 != j2:
                                G_discrete_outputs_batch[i, j1] = G_discrete_outputs_batch[i, j2]
                            j1 += 1
                            if j2 % 2 == 1:
                                val_increased_length += 1
                    G_discrete_length_batch[i] = j1

                BB_inputs_batch = np.concatenate((G_discrete_outputs_batch,
                                                  np.hstack((X_benign_batch, X_benign_batch))))
                BB_inputs_length_batch = np.concatenate((G_discrete_length_batch, benign_sequence_length_batch))
                BB_discriminator_result_batch = self.D.predict(BB_inputs_batch, BB_inputs_length_batch)
                X_benign_one_hot_batch = np.eye(self.num_tokens + 1)[np.hstack((X_benign_batch, X_benign_batch))]
                D_inputs_batch = np.concatenate((G_outputs_batch, X_benign_one_hot_batch))
                D_inputs_length_batch = np.concatenate((2 * sequence_length_batch, benign_sequence_length_batch))

                # get D loss
                D_loss_value = self.sess.run(self.D_loss, feed_dict={
                    self.D_all_inputs: D_inputs_batch,
                    self.D_all_inputs_length: D_inputs_length_batch,
                    self.discriminator_result: BB_discriminator_result_batch
                })
                val_D_loss += D_loss_value

                # get G loss
                G_loss_value, G_regu_value = self.sess.run([self.G_loss, self.G_regu], feed_dict={
                    self.inputs: X_batch,
                    self.inputs_length: sequence_length_batch,
                    self.Gumbel_samples: Gumbel_samples_batch
                })
                val_G_loss += G_loss_value
                val_G_regu += G_regu_value
                val_tpr += BB_discriminator_result_batch[:self.batch_size].mean()
            val_D_loss /= len(X_val) / self.batch_size
            val_G_loss /= len(X_val) / self.batch_size
            val_G_regu /= len(X_val) / self.batch_size
            val_tpr /= len(X_val) / self.batch_size
            val_increased_length /= len(X_val)

            if val_tpr < best_val_tpr:
                best_val_tpr = val_tpr
                best_val_epoch = epoch

            log_message = 'Epoch %d: Train Loss: D %g/ G %g/ regu %g/ len+ %g/ TPR %g. ' \
                          'Val Loss: D %g/ G %g/ regu %g/ len+ %g TPR %g' % \
                          (epoch, train_D_loss, train_G_loss, train_G_regu, train_increased_length, train_tpr,
                           val_D_loss, val_G_loss, val_G_regu, val_increased_length, val_tpr)
            print(log_message)
            with open(self.model_path + '--training_log.txt', 'a') as f:
                f.write(log_message + '\n')

            if epoch - best_val_epoch >= self.max_epoch_no_improvement:
                self.saver.restore(self.sess, self.model_path + '-' + str(best_val_epoch))
                break

    def sample(self, X, sequence_length):
        """ generate samples for X
        :param X: the list of data matrix(first element is malware, second element is benign)
                  the data matrix shape: [num examples, max sequence length].
        :param sequence_length: the list of length (first element is malware, second element is benign)
                  the length of each row of X. shape: [num examples]
        :return: the generated samples, the corresponding sequence_length
        """
        # get the benign data
        X_benign = X[1]
        X = X[0]
        benign_sequence_length = sequence_length[1]
        sequence_length = sequence_length[0]

        num_samples = len(X)
        X = np.concatenate((X, X[:self.batch_size - 1]))
        sequence_length = np.concatenate((sequence_length, sequence_length[:self.batch_size - 1]))
        generated_X = np.zeros((len(X), 2 * self.max_length), dtype=np.int32)
        generated_sequence_length = np.zeros((len(X),), dtype=np.int32)

        for start, end in zip(range(0, len(X), self.batch_size),
                              range(self.batch_size, len(X) + 1, self.batch_size)):
            X_batch = X[start: end]
            sequence_length_batch = sequence_length[start: end]
            Gumbel_samples_batch = 1e-10 + (1 - 2e-10) * \
                                           np.random.rand(self.batch_size, self.max_length, self.num_tokens + 1)
            Gumbel_samples_batch = -np.log(-np.log(Gumbel_samples_batch))
            # get the generated malware
            G_discrete_outputs_batch, G_outputs_batch = self.sess.run([self.G_discrete_outputs, self.G_outputs],
                                                                      feed_dict={
                                                                          self.inputs: X_batch,
                                                                          self.inputs_length: sequence_length_batch,
                                                                          self.Gumbel_samples: Gumbel_samples_batch
                                                                      })
            # remove non-output token
            G_discrete_length_batch = 2 * sequence_length_batch
            for i in range(self.batch_size):
                j1 = 0
                for j2 in range(G_discrete_length_batch[i]):
                    if G_discrete_outputs_batch[i, j2] < self.num_tokens:
                        if j1 != j2:
                            G_discrete_outputs_batch[i, j1] = G_discrete_outputs_batch[i, j2]
                        j1 += 1
                G_discrete_length_batch[i] = j1

            generated_X[start: end] = G_discrete_outputs_batch
            generated_sequence_length[start: end] = G_discrete_length_batch

        return generated_X[:num_samples], generated_sequence_length[:num_samples]