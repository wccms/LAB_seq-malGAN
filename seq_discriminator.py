from sklearn import metrics
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
from tensorflow import nn
import math

class SeqDiscriminator(object):
    """
    The basic class for sequence discriminator
    """
    def train(self, X, sequence_length, y):
        """ train the model
        :param X: the data matrix. shape: [num examples, max sequence length].
        :param sequence_length: the length of each row of X. shape: [num examples]
        :param y: the label. shape: [num examples]
        :return: a tuple of score_dict, the classification score_dict of training and val data.
        """
        raise NotImplementedError("Abstract method")

    def predict(self, X, sequence_length):
        """ predict the label of X
        :param X: the data matrix. shape: [num examples, max sequence length].
        :param sequence_length: the length of each row of X. shape: [num examples]
        :return: the label of X. shape: [num examples]
        """
        raise NotImplementedError("Abstract method")

    def evaluate(self, X, sequence_length, y):
        """ evaluate the classification performance of X with respect of y
        :param X: the test data. shape: [num examples, max sequence length].
        :param sequence_length: the length of each row of X. shape: [num examples]
        :param y: the ground truth label of X. shape: [num examples]
        :return: a dict of performance scores
        """
        raise NotImplementedError("Abstract method")

    def _score(self, true_label, predicted_prob):
        """ calculate the performance score for binary calssification
        :param true_label: the ground truth score
        :param predicted_label: the predicted probability
        :return: a dict of scores
        """
        score_dict = dict()
        score_dict['AUC'] = metrics.roc_auc_score(true_label, predicted_prob)
        predicted_label = [0 if prob < 0.5 else 1 for prob in predicted_prob]
        score_dict['Accuracy'] = metrics.accuracy_score(true_label, predicted_label)
        cm = metrics.confusion_matrix(true_label, predicted_label)
        score_dict['Confusion Matrix'] = cm
        score_dict['TPR'] = cm[1, 1] / float(cm[1, 0] + cm[1, 1])
        score_dict['FPR'] = cm[0, 1] / float(cm[0, 0] + cm[0, 1])
        return score_dict


class RNN_Classifier(SeqDiscriminator):
    def __init__(self, cell_type='LSTM', rnn_layers=[512], ff_layers=[512],
                 bidirectional=True, attention_layers=None, batch_size=128,
                 num_tokens=161, max_length=2048, num_classes=2,
                 max_epoch=1000, max_epoch_no_improvement=25, learning_rate=0.001,
                 model_path='model'):
        """
        :param cell_type: 'LSTM', 'RNN', 'GRU'
        :param rnn_layers: a list of integer
        :param ff_layers: a list of integer
        :param bidirectional: whether to use bidirectional RNN
        :param attention_layers: a list of integer or None, None means attention is not used
        :param batch_size: the size of mini-batch
        :param num_tokens: the number of distinct tokens of input data
        :param max_length: the maximum length of the input sequence
        :param num_classes: the number of classes
        :param max_epoch: the maximum epoch for training RNN
        :param max_epoch_no_improvement: if the performance of the val set doesn't improve for
                max_epoch_no_improvement, stop training
        :param learning_rate: the learning rate for RNN
        :param model_path: the path to save the model
        """
        self.cell_type = cell_type
        self.rnn_layers = rnn_layers
        self.ff_layers = ff_layers + [num_classes]
        self.bidirectional = bidirectional
        self.attention_layers = (attention_layers + [1]) if attention_layers is not None else None
        self.batch_size = batch_size
        self.num_tokens = num_tokens
        self.max_length = max_length
        self.num_classes = num_classes
        self.max_epoch = max_epoch
        self.max_epoch_no_improvement = max_epoch_no_improvement
        self.learning_rate = learning_rate
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

        # set the depth of cell
        if len(self.rnn_layers) == 1:
            cell = cell_element(self.rnn_layers[0])
        elif len(self.rnn_layers) > 1:
            cell_elements = []
            for rnn_layer in self.rnn_layers:
                cell_elements.append(cell_element(rnn_layer))
            cell = tf.contrib.rnn.MultiRNNCell(cell_elements)
        else:
            raise ValueError('rnn_layers must have at least one element')

        self.inputs = tf.placeholder(tf.int32, [self.batch_size, self.max_length])
        self.inputs_length = tf.placeholder(tf.int32, [self.batch_size])
        self.outputs = tf.placeholder(tf.int32, [self.batch_size])
        inputs_one_hot = tf.one_hot(self.inputs, self.num_tokens)

        # bidirectional
        if self.bidirectional:
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell, cell, inputs_one_hot,
                                                         self.inputs_length, dtype=tf.float32,
                                                         swap_memory=True, time_major=False)
            output_fw, output_bw = outputs
            per_step_outputs = tf.concat(outputs, 2)
            if self.attention_layers is None:
                output_fw = tf.reverse_sequence(output_fw, self.inputs_length, seq_dim=1, batch_dim=0)
                ff_input = tf.concat([output_fw[:, 0, :], output_bw[:, 0, :]], 1)
            else:
                attention_input = tf.concat([output_fw, output_bw], 2)
        else:
            outputs, _ = tf.nn.dynamic_rnn(cell, inputs_one_hot, self.inputs_length,
                                           dtype=tf.float32, swap_memory=True, time_major=False)
            per_step_outputs = outputs
            if self.attention_layers is None:
                outputs = tf.reverse_sequence(outputs, self.inputs_length, seq_dim=1, batch_dim=0)
                ff_input = outputs[:, 0, :]
            else:
                attention_input = outputs

        # attention
        if self.attention_layers is not None:
            last_attention_hidden = self.rnn_layers[-1] * (2 if self.bidirectional else 1)
            attention_hidden = tf.reshape(attention_input, [self.batch_size * self.max_length, last_attention_hidden])
            for layer, attention_layer in enumerate(self.attention_layers):
                W_attention = tf.Variable(tf.random_uniform([last_attention_hidden, attention_layer], -0.1, 0.1),
                                          name='W_attention_%d' % (layer,))
                last_attention_hidden = attention_layer
                b_attention = tf.Variable(tf.zeros([attention_layer]), name='b_attention_%d' % (layer,))
                attention_hidden = tf.matmul(attention_hidden, W_attention) + b_attention
                if layer < len(self.attention_layers) - 1:
                    attention_hidden = tf.nn.tanh(attention_hidden)
            attention_weights = tf.exp(tf.reshape(attention_hidden, [self.batch_size, self.max_length]))
            inputs_mask = tf.sequence_mask(self.inputs_length, self.max_length, dtype=tf.float32)
            attention_weights *= inputs_mask
            attention_weights_sum = tf.reduce_sum(attention_weights, 1, keep_dims=True)
            attention_weights /= attention_weights_sum
            ff_input = tf.reduce_sum(tf.multiply(attention_input, tf.expand_dims(attention_weights, 2)), 1)

        # feed forwards
        last_ff_hidden = self.rnn_layers[-1] * (2 if self.bidirectional else 1)
        ff_hidden = ff_input
        params_ff = []
        for layer, ff_layer in enumerate(self.ff_layers):
            W_ff = tf.Variable(tf.random_uniform([last_ff_hidden, ff_layer], -0.1, 0.1),
                               name='W_ff_%d' % (layer,))
            last_ff_hidden = ff_layer
            b_ff = tf.Variable(tf.zeros([ff_layer]), name='b_ff_%d' % (layer,))
            ff_hidden = tf.matmul(ff_hidden, W_ff) + b_ff
            params_ff.append((W_ff, b_ff))
            if layer < len(self.ff_layers) - 1:
                ff_hidden = tf.nn.tanh(ff_hidden)

        self.probability = tf.nn.softmax(ff_hidden)
        self.loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.outputs, logits=ff_hidden))

        opt = tf.train.AdamOptimizer(self.learning_rate)
        grads_and_vars = opt.compute_gradients(self.loss)
        grads_and_vars = [(tf.clip_by_value(gv[0], -1.0, 1.0), gv[1]) for gv in grads_and_vars]
        self.train_op = opt.apply_gradients(grads_and_vars)
        self.init_op = tf.global_variables_initializer()

        # classify the sequence per step
        per_step_hidden = tf.reshape(per_step_outputs, [self.batch_size * self.max_length,
                                                        self.rnn_layers[-1] * (2 if self.bidirectional else 1)])
        for layer in range(len(params_ff)):
            W_ff, b_ff = params_ff[layer]
            per_step_hidden = tf.matmul(per_step_hidden, W_ff) + b_ff
            if layer < len(self.ff_layers) - 1:
                per_step_hidden = tf.nn.tanh(per_step_hidden)
        self.per_step_result = tf.reshape(tf.argmax(per_step_hidden, axis=1), [self.batch_size, self.max_length])

    def train(self, X, sequence_length, y):
        """ train the model
        :param X: the data matrix. shape: [num examples, max sequence length].
        :param sequence_length: the length of each row of X. shape: [num examples]
        :param y: the label. shape: [num examples]
        :return: a tuple of score_dict, the classification score_dict of training and val data.
        """
        self.sess.run(self.init_op)
        # shuffle and split train and val data
        index = np.arange(len(X))
        np.random.shuffle(index)
        X = X[index]
        sequence_length = sequence_length[index]
        y = y[index]
        num_training_samples = int(len(X) * 0.75)
        X_val = X[num_training_samples:]
        sequence_length_val = sequence_length[num_training_samples:]
        y_val = y[num_training_samples:]
        X = X[:num_training_samples]
        sequence_length = sequence_length[:num_training_samples]
        y = y[:num_training_samples]
        # self.saver.restore(self.sess, '../D_models/20171104_birnn_LSTM_128_ff_128_attention_None_max_len_1024_lr_0.001/model-24')
        # return self.evaluate(X, sequence_length, y), self.evaluate(X_val, sequence_length_val, y_val)

        best_val_loss = 1000.0
        best_val_epoch = 0
        for epoch in range(self.max_epoch):
            train_loss = 0.0
            for start, end in zip(range(0, len(X), self.batch_size),
                                  range(self.batch_size, len(X) + 1, self.batch_size)):
                X_batch = X[start: end]
                sequence_length_batch = sequence_length[start: end]
                y_batch = y[start: end]
                _, loss_value = self.sess.run([self.train_op, self.loss], feed_dict={
                    self.inputs: X_batch,
                    self.inputs_length: sequence_length_batch,
                    self.outputs: y_batch
                })
                train_loss += loss_value
            train_loss /= len(X) / self.batch_size
            self.saver.save(self.sess, self.model_path, global_step=epoch)

            val_loss = 0.0
            for start, end in zip(range(0, len(X_val), self.batch_size),
                                  range(self.batch_size, len(X_val) + 1, self.batch_size)):
                X_batch = X_val[start: end]
                sequence_length_batch = sequence_length_val[start: end]
                y_batch = y_val[start: end]
                loss_value = self.sess.run(self.loss, feed_dict={
                    self.inputs: X_batch,
                    self.inputs_length: sequence_length_batch,
                    self.outputs: y_batch
                })
                val_loss += loss_value
            val_loss /= len(X_val) / self.batch_size

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_epoch = epoch

            log_message = 'Discriminator Epoch %d: Loss: Training=%g, Val=%g' % (epoch, train_loss, val_loss)
            print(log_message)
            with open(self.model_path + '--training_log.txt', 'a') as f:
                f.write(log_message + '\n')

            if epoch - best_val_epoch >= self.max_epoch_no_improvement:
                self.saver.restore(self.sess, self.model_path + '-' + str(best_val_epoch))
                break

        return self.evaluate(X, sequence_length, y), self.evaluate(X_val, sequence_length_val, y_val)

    def _predict_proba(self, X, sequence_length):
        """ predict the probability distribution of X
        :param X: the data matrix. shape: [num examples, max sequence length].
        :param sequence_length: the length of each row of X. shape: [num examples]
        :return: the probability distribution of X. shape: [num examples, num classes]
        """
        num_samples = len(X)
        if num_samples < self.batch_size:
            X = np.tile(X, (self.batch_size / num_samples + 1, 1))[:self.batch_size]
            sequence_length = np.tile(sequence_length, (self.batch_size / num_samples + 1, 1))[:self.batch_size]
        else:
            X = np.concatenate((X, X[:self.batch_size - 1]))
            sequence_length = np.concatenate((sequence_length, sequence_length[:self.batch_size - 1]))
        proba = np.zeros((len(X), self.num_classes), dtype=np.float32)
        for start, end in zip(range(0, len(X), self.batch_size),
                              range(self.batch_size, len(X) + 1, self.batch_size)):
            X_batch = X[start: end]
            sequence_length_batch = sequence_length[start: end]
            proba_batch = self.sess.run(self.probability, feed_dict={
                self.inputs: X_batch,
                self.inputs_length: sequence_length_batch,
            })
            proba[start: end] = proba_batch

        return proba[:num_samples]

    def predict(self, X, sequence_length):
        """ predict the label of X
        :param X: the data matrix. shape: [num examples, max sequence length].
        :param sequence_length: the length of each row of X. shape: [num examples]
        :return: the label of X. shape: [num examples]
        """
        return np.argmax(self._predict_proba(X, sequence_length), axis=1)

    def evaluate(self, X, sequence_length, y):
        """ evaluate the classification performance of X with respect of y
        :param X: the test data. shape: [num examples, max sequence length].
        :param sequence_length: the length of each row of X.\. shape: [num examples]
        :param y: the ground truth label of X. shape: [num examples]
        :return: a dict of performance scores
        """
        return self._score(y, self._predict_proba(X, sequence_length)[:, 1])

    def predict_per_step(self, X, sequence_length):
        """ predict the label of X per step
        :param X: the data matrix. shape: [num examples, max sequence length].
        :param sequence_length: the length of each row of X. shape: [num examples]
        :return: the label of X[:, 0], X[:, 0:1], X[:, 0:2], ..., X[:, 0:T-1].
                 shape: [num examples, max sequence length]
        """
        num_samples = len(X)
        if num_samples < self.batch_size:
            X = np.tile(X, (self.batch_size / num_samples + 1, 1))[:self.batch_size]
            sequence_length = np.tile(sequence_length, (self.batch_size / num_samples + 1, 1))[:self.batch_size]
        else:
            X = np.concatenate((X, X[:self.batch_size - 1]))
            sequence_length = np.concatenate((sequence_length, sequence_length[:self.batch_size - 1]))
        result = np.zeros((len(X), self.max_length), dtype=np.int32)
        for start, end in zip(range(0, len(X), self.batch_size),
                              range(self.batch_size, len(X) + 1, self.batch_size)):
            X_batch = X[start: end]
            sequence_length_batch = sequence_length[start: end]
            result_batch = self.sess.run(self.per_step_result, feed_dict={
                self.inputs: X_batch,
                self.inputs_length: sequence_length_batch,
            })
            result[start: end] = result_batch

        return result[:num_samples]


class RNN_LM(SeqDiscriminator):
    def __init__(self, cell_type='LSTM', rnn_layers=[512], batch_size=128,
                 num_tokens=161, max_length=2048, num_classes=2,
                 max_epoch=1000, max_epoch_no_improvement=25, learning_rate=0.001,
                 model_path='model'):
        """
        :param cell_type: 'LSTM', 'RNN', 'GRU'
        :param rnn_layers: a list of integer
        :param batch_size: the size of mini-batch
        :param num_tokens: the number of distinct tokens of input data
        :param max_length: the maximum length of the input sequence
        :param num_classes: the number of classes
        :param max_epoch: the maximum epoch for training RNN
        :param max_epoch_no_improvement: if the performance of the val set doesn't improve for
                max_epoch_no_improvement, stop training
        :param learning_rate: the learning rate for RNN
        :param model_path: the path to save the model
        """
        self.model = None
        self.cell_type = cell_type
        self.rnn_layers = rnn_layers
        self.batch_size = batch_size
        self.num_tokens = num_tokens
        self.max_length = max_length
        self.num_classes = num_classes
        self.max_epoch = max_epoch
        self.max_epoch_no_improvement = max_epoch_no_improvement
        self.learning_rate = learning_rate
        self.model_path = model_path
        self.model = None
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

        # set the depth of cell
        if len(self.rnn_layers) == 1:
            cell = cell_element(self.rnn_layers[0])
        elif len(self.rnn_layers) > 1:
            cell_elements = []
            for rnn_layer in self.rnn_layers:
                cell_elements.append(cell_element(rnn_layer))
            cell = tf.nn.rnn_cell.MultiRNNCell(cell_elements)
        self.inputs = tf.placeholder(tf.int32, [self.batch_size, self.max_length])
        self.inputs_length = tf.placeholder(tf.int32, [self.batch_size])
        previous_tokens = self.inputs[:, :-1]
        next_tokens = self.inputs[:, 1:]
        tokens_length = self.inputs_length - 1
        previous_tokens_one_hot = tf.one_hot(previous_tokens, self.num_tokens)

        with tf.variable_scope("RNN_LM"):
            # training
            outputs, _ = tf.nn.dynamic_rnn(cell, previous_tokens_one_hot, tokens_length,
                                           dtype=tf.float32, swap_memory=True, time_major=False)

            # extracting features
            tf.get_variable_scope().reuse_variables()
            outputs_test, _ = tf.nn.dynamic_rnn(cell, tf.one_hot(self.inputs, self.num_tokens), self.inputs_length,
                                                dtype=tf.float32, swap_memory=True, time_major=False)

            last_state = tf.reverse_sequence(outputs_test, self.inputs_length, seq_dim=1, batch_dim=0)[:, 0, :]
            max_pooling = tf.reduce_max(outputs_test, 1)
            self.features = tf.concat([last_state, max_pooling], 1)

        outputs = tf.reshape(outputs, [self.batch_size * (self.max_length - 1), self.rnn_layers[-1]])
        W_output= tf.Variable(tf.random_uniform([self.rnn_layers[-1], self.num_tokens], -0.1, 0.1), name='W_output')
        b_output = tf.Variable(tf.zeros([self.num_tokens]), name='b_output')
        outputs = tf.matmul(outputs, W_output) + b_output

        next_tokens = tf.reshape(next_tokens, [self.batch_size * (self.max_length - 1)])
        self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(outputs, next_tokens)
        tokens_mask = tf.sequence_mask(tokens_length, self.max_length - 1, dtype=tf.float32)
        self.loss = tf.multiply(self.loss, tf.reshape(tokens_mask, [self.batch_size * (self.max_length - 1)]))
        self.loss = tf.reduce_sum(self.loss) / self.batch_size

        opt = tf.train.AdamOptimizer(self.learning_rate)
        grads_and_vars = opt.compute_gradients(self.loss)
        grads_and_vars = [(tf.clip_by_value(gv[0], -1.0, 1.0), gv[1]) for gv in grads_and_vars]
        self.train_op = opt.apply_gradients(grads_and_vars)
        self.init_op = tf.global_variables_initializer()

    def _extract_feature(self, X, sequence_length):
        """ extract feature from X
        :param X: the data matrix. shape: [num examples, max sequence length].
        :param sequence_length: the length of each row of X. shape: [num examples]
        :return: the feature of X. shape: [num examples, num features]
        """
        num_samples = len(X)
        X = np.concatenate((X, X[:self.batch_size - 1]))
        sequence_length = np.concatenate((sequence_length, sequence_length[:self.batch_size - 1]))
        feature_vector = np.zeros((len(X), 2 * self.rnn_layers[-1]), dtype=np.float32)
        for start, end in zip(range(0, len(X), self.batch_size),
                              range(self.batch_size, len(X) + 1, self.batch_size)):
            X_batch = X[start: end]
            sequence_length_batch = sequence_length[start: end]
            feature_batch = self.sess.run(self.features, feed_dict={
                self.inputs: X_batch,
                self.inputs_length: sequence_length_batch,
            })
            feature_vector[start: end] = feature_batch

        return feature_vector[:num_samples]

    def train(self, X, sequence_length, y):
        """ train the model
        :param X: the data matrix. shape: [num examples, max sequence length].
        :param sequence_length: the length of each row of X. shape: [num examples]
        :param y: the label. shape: [num examples]
        :return: a tuple of score_dict, the classification score_dict of training and val data.
        """
        self.sess.run(self.init_op)
        # shuffle and split train and val data
        index = np.arange(len(X))
        np.random.shuffle(index)
        X = X[index]
        sequence_length = sequence_length[index]
        y = y[index]
        num_training_samples = int(len(X) * 0.75)
        X_val = X[num_training_samples:]
        sequence_length_val = sequence_length[num_training_samples:]
        y_val = y[num_training_samples:]
        X = X[:num_training_samples]
        sequence_length = sequence_length[:num_training_samples]
        y = y[:num_training_samples]

        best_val_loss = 1000.0
        best_val_epoch = 0
        for epoch in range(self.max_epoch):
            train_loss = 0.0
            for start, end in zip(range(0, len(X), self.batch_size),
                                  range(self.batch_size, len(X) + 1, self.batch_size)):
                X_batch = X[start: end]
                sequence_length_batch = sequence_length[start: end]
                _, loss_value = self.sess.run([self.train_op, self.loss], feed_dict={
                    self.inputs: X_batch,
                    self.inputs_length: sequence_length_batch
                })
                train_loss += loss_value
            train_loss /= len(X) / self.batch_size
            self.saver.save(self.sess, self.model_path, global_step=epoch)

            val_loss = 0.0
            for start, end in zip(range(0, len(X_val), self.batch_size),
                                  range(self.batch_size, len(X_val) + 1, self.batch_size)):
                X_batch = X_val[start: end]
                sequence_length_batch = sequence_length_val[start: end]
                loss_value = self.sess.run(self.loss, feed_dict={
                    self.inputs: X_batch,
                    self.inputs_length: sequence_length_batch
                })
                val_loss += loss_value
            val_loss /= len(X_val) / self.batch_size

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_epoch = epoch

            log_message = 'Discriminator Epoch %d: Loss: Training=%g, Val=%g' % (epoch, train_loss, val_loss)
            print(log_message)
            with open(self.model_path + '--training_log.txt', 'a') as f:
                f.write(log_message + '\n')

            if epoch - best_val_epoch >= self.max_epoch_no_improvement:
                self.saver.restore(self.sess, self.model_path + '-' + str(best_val_epoch))
                break

        X_feature = self._extract_feature(X, sequence_length)
        self.model = RandomForestClassifier(n_estimators=100, n_jobs=10)
        self.model.fit(X_feature, y)

        return self.evaluate(X, sequence_length, y), self.evaluate(X_val, sequence_length_val, y_val)

    def predict(self, X, sequence_length):
        """ predict the label of X
        :param X: the data matrix. shape: [num examples, max sequence length].
        :param sequence_length: the length of each row of X. shape: [num examples]
        :return: the label of X. shape: [num examples]
        """
        X_feature = self._extract_feature(X, sequence_length)
        return self.model.predict(X_feature)

    def evaluate(self, X, sequence_length, y):
        """ evaluate the classification performance of X with respect of y
        :param X: the test data. shape: [num examples, max sequence length].
        :param sequence_length: the length of each row of X.\. shape: [num examples]
        :param y: the ground truth label of X. shape: [num examples]
        :return: a dict of performance scores
        """
        X_feature = self._extract_feature(X, sequence_length)
        return self._score(y, self.model.predict_proba(X_feature)[:, 1])


class NGram(SeqDiscriminator):
    def __init__(self, N=2, num_features=2000, max_length=2048):
        self.N = N
        self.num_features = num_features
        self.features = dict()  # tuple of ngram to index, e.g. {(0,1,2):1}
        self.max_length = max_length
        self.model = None

    def _select_feature(self, X, sequence_length, y):
        """ select n-gram feature
        :param X: the data matrix. shape: [num examples, max sequence length].
        :param sequence_length: the length of each row of X. shape: [num examples]
        :param y: the label. shape: [num examples]
        """
        num_classes = max(y) + 1
        num_samples = len(X)
        num_samples_per_class = []
        for class_value in range(num_classes):
            num_samples_per_class.append((y == class_value).sum())
        ngram_class = []
        for i in range(len(X)):
            ngrams = []
            for j in range(sequence_length[i] - self.N):
                ngrams.append((tuple(X[i, j:j+self.N]), y[i]))
            ngram_class.extend(list(set(ngrams)))    # list of ((f1, f2,..., fn), c)
        ngram_2_class_count = dict()    # (f1, f2,..., fn) --> [n1, n2, n3]
        for ngram, class_value in ngram_class:
            if ngram in ngram_2_class_count:
                ngram_2_class_count[ngram][class_value] += 1
            else:
                ngram_2_class_count[ngram] = [0] * num_classes
                ngram_2_class_count[ngram][class_value] = 1

        def info_gain(class_count):
            ig = 0.0
            prob_Vj = float(sum(class_count)) / num_samples
            prob_Vj_r = 1.0 - prob_Vj
            for class_idx, file_count_in_class in enumerate(num_samples_per_class):
                # presence of the ngram
                num_Vj_C = class_count[class_idx]
                # absence of the ngram (_r)
                num_Vj_r_C = file_count_in_class - num_Vj_C
                for num_V_C, prob_V in [(num_Vj_C, prob_Vj), (num_Vj_r_C, prob_Vj_r)]:
                    if num_V_C > 0:
                        ig += float(num_V_C) / num_samples \
                              * math.log(num_V_C / prob_V / file_count_in_class) / math.log(2.0)
            return ig

        # calculate the information gain
        ngram_ig = []
        for ngram, class_count in ngram_2_class_count.iteritems():
            ngram_ig.append((ngram, info_gain(class_count)))
        ngram_ig.sort(key=lambda x: -x[1])
        self.features = dict()
        for i in range(min(self.num_features, len(ngram_ig))):
            self.features[ngram_ig[i][0]] = i

    def _extract_feature(self, X, sequence_length):
        """ extract n-gram feature
        :param X: the data matrix. shape: [num examples, max sequence length].
        :param sequence_length: the length of each row of X. shape: [num examples]
        :return: the feature matrix. shape: [num examples, num features]
        """
        X_feature = np.zeros((len(X), len(self.features)), dtype=np.int32)
        for i in range(len(X)):
            for j in range(sequence_length[i] - self.N):
                ngram = tuple(X[i, j:j+self.N])
                if ngram in self.features:
                    X_feature[i, self.features[ngram]] = 1
        return X_feature

    def train(self, X, sequence_length, y):
        """ train the model
        :param X: the data matrix. shape: [num examples, max sequence length].
        :param sequence_length: the length of each row of X. shape: [num examples]
        :param y: the label. shape: [num examples]
        :return: a tuple of score_dict, the classification score_dict of training and val data.
        """
        sequence_length = np.minimum(sequence_length, self.max_length)
        # shuffle and split train and val data
        index = np.arange(len(X))
        np.random.shuffle(index)
        X = X[index]
        sequence_length = sequence_length[index]
        y = y[index]
        num_training_samples = int(len(X) * 0.75)
        X_val = X[num_training_samples:]
        sequence_length_val = sequence_length[num_training_samples:]
        y_val = y[num_training_samples:]
        X = X[:num_training_samples]
        sequence_length = sequence_length[:num_training_samples]
        y = y[:num_training_samples]

        self._select_feature(X, sequence_length, y)
        X_feature = self._extract_feature(X, sequence_length)
        self.model = RandomForestClassifier(n_estimators=100, n_jobs=10)
        self.model.fit(X_feature, y)

        return self.evaluate(X, sequence_length, y), self.evaluate(X_val, sequence_length_val, y_val)

    def predict(self, X, sequence_length):
        """ predict the label of X
        :param X: the data matrix. shape: [num examples, max sequence length].
        :param sequence_length: the length of each row of X. shape: [num examples]
        :return: the label of X. shape: [num examples]
        """
        sequence_length = np.minimum(sequence_length, self.max_length)
        X_feature = self._extract_feature(X, sequence_length)
        return self.model.predict(X_feature)

    def evaluate(self, X, sequence_length, y):
        """ evaluate the classification performance of X with respect of y
        :param X: the test data. shape: [num examples, max sequence length].
        :param sequence_length: the length of each row of X.\. shape: [num examples]
        :param y: the ground truth label of X. shape: [num examples]
        :return: a dict of performance scores
        """
        sequence_length = np.minimum(sequence_length, self.max_length)
        X_feature = self._extract_feature(X, sequence_length)
        return self._score(y, self.model.predict_proba(X_feature)[:, 1])
