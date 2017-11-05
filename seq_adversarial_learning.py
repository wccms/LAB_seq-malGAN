import numpy as np
from seq_discriminator import RNN_Classifier
from seq_discriminator import RNN_LM
from seq_discriminator import NGram
from seq_generator import SeqMerger
from seq_generator import SeqMergerA3C
from seq_generator import SeqInserter
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from datetime import datetime
import sys
import shutil

def load_dataset(data_path, max_length=2048, pad_length=2048):
    """ load the dataset
    :param data_path: the path of the dataset
    :param max_length: the max length of a sequence
    :param pad_length: the padded length of the returned matrix
    :return: X_malware, malware_length, X_benign, benign_length
            X_malware and X_benign are the matrix of malware and benign.
                shape: [num instances, max_length + pad_length]
            malware_length and benign_length are the length of the matrix, which should be smaller than max_length
                shape: [num instances]
    """
    X_malware = []
    malware_length = []
    X_benign = []
    benign_length = []
    for line in open(data_path):
        elements = line.strip().split(';')
        Xi = []
        for element in elements[2:-1]:
            if len(element) <= 0:
                continue
            for digit in element.split(','):
                Xi.append(int(digit))
        Xi = Xi[:max_length]
        if elements[1] is '0':
            benign_length.append(len(Xi))
            X_benign.append(np.array(Xi + [0] * (max_length + pad_length - len(Xi)), dtype=np.int32))
        else:
            malware_length.append(len(Xi))
            X_malware.append(np.array(Xi + [0] * (max_length + pad_length - len(Xi)), dtype=np.int32))
    return np.vstack(X_malware), np.array(malware_length), np.vstack(X_benign), np.array(benign_length)


def _get_rnn_model(model_path='model', max_length=2048, rnn_layers=[128], ff_layers=[128],
                   bidirectional=True, attention_layers=None, learning_rate=0.001):
    model = RNN_Classifier(cell_type='LSTM', rnn_layers=rnn_layers, ff_layers=ff_layers,
                            bidirectional=bidirectional, attention_layers=attention_layers, batch_size=128,
                            num_tokens=161, max_length=max_length, num_classes=2,
                            max_epoch=100, max_epoch_no_improvement=5, learning_rate=learning_rate,
                            model_path=model_path)
    return model


def _get_rnnlm_model(model_path='model', max_length=2048):
    model = RNN_LM(cell_type='LSTM', rnn_layers=[128], batch_size=128,
                    num_tokens=161, max_length=max_length, num_classes=2,
                    max_epoch=100, max_epoch_no_improvement=5, learning_rate=0.001,
                    model_path=model_path)
    return model


def _get_ngram_model(model_path='model', max_length=2048):
    model = NGram(N=2, num_features=2000, max_length=max_length)
    return model


def tune_discriminator_parameters():
    X_malware, malware_length, X_benign, benign_length = load_dataset('../data/API_rand_trainval_len_2048.txt', 1024, 0)
    X = np.vstack((X_malware, X_benign))
    sequence_length = np.hstack((malware_length, benign_length))
    y = np.array([1] * len(X_malware) + [0] * len(X_benign))
    #X_malware_test, malware_length_test, X_benign_test, benign_length_test = \
    #    load_dataset('', 2048, 2048)
    #X_test = np.vstack((X_malware_test, X_benign_test))
    #test_sequence_length = np.hstack((malware_length_test, benign_length_test))
    #y_test = np.array([1] * len(X_malware_test) + [0] * len(X_benign_test))
    rnn_layers = [128]
    ff_layers = [128]
    bidirectional = False
    attention_layers = None
    max_length = 1024
    learning_rate = 0.001
    tag = '20171104_%s_LSTM_%s_ff_%s_attention_%s_maxLen_%d_lr_%g' % (
        'birnn' if bidirectional else 'rnn',
        '_'.join([str(layer) for layer in rnn_layers]),
        '_'.join([str(layer) for layer in ff_layers]),
        '_'.join([str(layer) for layer in attention_layers]) if attention_layers is not None else 'None',
        max_length,
        learning_rate
    )
    dir_path = '../D_models/' + tag
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    if os.path.exists(os.path.join(dir_path, 'code')):
        shutil.rmtree(os.path.join(dir_path, 'code'))
    shutil.copytree('.', os.path.join(dir_path, 'code'))
    model_path = dir_path + '/model'
    log_path = dir_path + '/log.txt'
    score_template = 'TPR %(TPR)f\tFPR %(FPR)f\tAccuracy %(Accuracy)f\tAUC %(AUC)f'

    model = _get_rnn_model(model_path, max_length=max_length, rnn_layers=rnn_layers, ff_layers=ff_layers,
                   bidirectional=bidirectional, attention_layers=attention_layers, learning_rate=learning_rate)
    #model = _get_rnnlm_model('../discriminator_models/rnnlm/0/model')
    #model = _get_ngram_model()
    train_result, val_result = model.train(X, sequence_length, y)
    log_message = 'Train result: ' + score_template % train_result + '\n'
    
    
    
    
    log_message += 'Val result: ' + score_template % val_result + '\n'
    print(log_message)
    with open(log_path, 'a') as f:
        f.write(log_message)


def learning_SeqMerger():
    cell_type = 'LSTM'
    encoder_layers = []
    encoder_bidirectional = True
    encoder_share_weights = True
    merge_layers = [128]
    batch_size = 2048
    noise_dim = 16
    num_samples = 8
    max_length = 100
    max_epoch = 1000
    max_epoch_no_improvement = 50
    baseline = True
    learning_rate = 0.01

    tag = '20170103_rnn_%s_encoder_%s_%s_%s_merger_%s_max_len_%d_noise_%d_samples_%d_%s_lr_%g' % (
        cell_type,
        '_'.join([str(layer) for layer in encoder_layers]),
        'birnn' if encoder_bidirectional else 'rnn',
        'share' if encoder_share_weights else 'no_share',
        '_'.join([str(layer) for layer in merge_layers]),
        max_length,
        noise_dim,
        num_samples,
        'baseline' if baseline else 'no_baseline',
        learning_rate
    )
    dir_path = '../seq_models/' + tag
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    if os.path.exists(os.path.join(dir_path, 'code')):
        shutil.rmtree(os.path.join(dir_path, 'code'))
    shutil.copytree('.', os.path.join(dir_path, 'code'))

    model_path = dir_path + '/model'
    log_path = dir_path + '/log.txt'
    score_template = 'TPR %(TPR)f\tFPR %(FPR)f\tAccuracy %(Accuracy)f\tAUC %(AUC)f'
    D = _get_rnn_model(dir_path + '/seq_discriminator_model', max_length=2 * max_length)
    G = SeqMerger(D, cell_type=cell_type, encoder_layers=encoder_layers, encoder_bidirectional=encoder_bidirectional,
                  encoder_share_weights=encoder_share_weights, merge_layers=merge_layers, batch_size=batch_size,
                  noise_dim=noise_dim, num_tokens=161, max_length=max_length, max_epoch=max_epoch,
                  max_epoch_no_improvement=max_epoch_no_improvement, num_samples=num_samples,
                  learning_rate=learning_rate, baseline=baseline, model_path=model_path)

    # load the data
    X_malware, malware_length, X_benign, benign_length = \
        load_dataset('../data/API_rand_trainval_len_2048.txt', max_length, 0)
    X = np.vstack((X_malware, X_benign))
    sequence_length = np.hstack((malware_length, benign_length))
    y = np.array([1] * len(X_malware) + [0] * len(X_benign))
    X_malware_test, malware_length_test, X_benign_test, benign_length_test = \
        load_dataset('../data/API_rand_test_len_2048.txt', max_length, 0)
    X_test = np.vstack((X_malware_test, X_benign_test))
    test_sequence_length = np.hstack((malware_length_test, benign_length_test))
    y_test = np.array([1] * len(X_malware_test) + [0] * len(X_benign_test))

    log_message = str(datetime.now()) + '\tTraining discriminative model on original dataset\n'
    D.train(np.hstack((X, np.zeros_like(X))), sequence_length, y)
    log_message += str(datetime.now()) + '\tTraining set result\t'
    log_message += score_template % D.evaluate(np.hstack((X, np.zeros_like(X))), sequence_length, y)
    log_message += '\n' + str(datetime.now()) + '\tTest set result\t'
    log_message += score_template % D.evaluate(np.hstack((X_test, np.zeros_like(X_test))), test_sequence_length, y_test)
    with open(log_path, 'a') as f:
        f.write(log_message + '\n')

    for i in range(50):
        log_message = str(datetime.now()) + '\tTraining generative model for the %d-th time\n' % (i,)
        #G.train(training_data_malware[:, :-1])
        G.train((X_malware, X_benign), (malware_length, benign_length))
        log_message += str(datetime.now()) + '\tGenerating examples\n'
        generated_training_malware, generated_training_malware_length = \
            G.sample((X_malware, X_benign), (malware_length, benign_length))
        generated_training_data = \
            np.vstack((generated_training_malware, np.hstack((X_benign, np.zeros_like(X_benign)))))
        generated_sequence_length = np.hstack((generated_training_malware_length, benign_length))
        generated_test_malware, generated_test_malware_length = \
            G.sample((X_malware_test, X_benign), (malware_length_test, benign_length))
        generated_test_data = \
            np.vstack((generated_test_malware, np.hstack((X_benign_test, np.zeros_like(X_benign_test)))))
        generated_sequence_length_test = np.hstack((generated_test_malware_length, benign_length_test))

        log_message += str(datetime.now()) + '\tTraining set result before re-training\t'
        log_message += score_template % D.evaluate(generated_training_data, generated_sequence_length, y)
        log_message += '\n' + str(datetime.now()) + '\tTest set result before re-training\t'
        log_message += score_template % D.evaluate(generated_test_data, generated_sequence_length_test, y_test)
        log_message += '\n' + str(datetime.now()) + '\tRe-training discriminative model\n'
        D.train(generated_training_data, generated_sequence_length, y)
        log_message += str(datetime.now()) + '\tTraining set result after re-training\t'
        log_message += score_template % D.evaluate(generated_training_data, generated_sequence_length, y)
        log_message += '\n' + str(datetime.now()) + '\tTest set result after re-training\t'
        log_message += score_template % D.evaluate(generated_test_data, generated_sequence_length_test, y_test)
        with open(log_path, 'a') as f:
            f.write(log_message + '\n\n')

def learning_SeqMergerA3C():
    cell_type = 'LSTM'
    encoder_layers = []
    encoder_bidirectional = True
    encoder_share_weights = True
    merge_layers = [128]
    value_layers = [128]
    batch_size = 256
    gamma = 0.99
    eps = 0.01
    noise_dim = -1
    max_length = 100
    max_epoch = 1000
    max_epoch_no_improvement = 50
    learning_rate = 0.001

    tag = '20170106_rnn_%s_encoder_%s_%s_%s_merger_%s_value_%s_batch_%d_gamma_%g_eps_%g_maxLen_%d_noise_%d_lr_%g' % (
        cell_type,
        '_'.join([str(layer) for layer in encoder_layers]),
        'birnn' if encoder_bidirectional else 'rnn',
        'share' if encoder_share_weights else 'no_share',
        '_'.join([str(layer) for layer in merge_layers]),
        '_'.join([str(layer) for layer in value_layers]),
        batch_size,
        gamma,
        eps,
        max_length,
        noise_dim,
        learning_rate
    )
    dir_path = '../seq_models/' + tag
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    if os.path.exists(os.path.join(dir_path, 'code')):
        shutil.rmtree(os.path.join(dir_path, 'code'))
    shutil.copytree('.', os.path.join(dir_path, 'code'))

    model_path = dir_path + '/model'
    log_path = dir_path + '/log.txt'
    score_template = 'TPR %(TPR)f\tFPR %(FPR)f\tAccuracy %(Accuracy)f\tAUC %(AUC)f'
    D = _get_rnn_model(dir_path + '/seq_discriminator_model', max_length=2 * max_length)
    G = SeqMergerA3C(D, cell_type=cell_type, encoder_layers=encoder_layers, encoder_bidirectional=encoder_bidirectional,
                  encoder_share_weights=encoder_share_weights, merge_layers=merge_layers, value_layers=value_layers,
                  batch_size=batch_size, gamma=gamma, eps=eps, noise_dim=noise_dim, num_tokens=161,
                  max_length=max_length, max_epoch=max_epoch, max_epoch_no_improvement=max_epoch_no_improvement,
                  learning_rate=learning_rate, model_path=model_path)

    # load the data
    X_malware, malware_length, X_benign, benign_length = \
        load_dataset('../data/API_rand_trainval_len_2048.txt', max_length, 0)
    X = np.vstack((X_malware, X_benign))
    sequence_length = np.hstack((malware_length, benign_length))
    y = np.array([1] * len(X_malware) + [0] * len(X_benign))
    X_malware_test, malware_length_test, X_benign_test, benign_length_test = \
        load_dataset('../data/API_rand_test_len_2048.txt', max_length, 0)
    X_test = np.vstack((X_malware_test, X_benign_test))
    test_sequence_length = np.hstack((malware_length_test, benign_length_test))
    y_test = np.array([1] * len(X_malware_test) + [0] * len(X_benign_test))

    log_message = str(datetime.now()) + '\tTraining discriminative model on original dataset\n'
    D.train(np.hstack((X, np.zeros_like(X))), sequence_length, y)
    log_message += str(datetime.now()) + '\tTraining set result\t'
    log_message += score_template % D.evaluate(np.hstack((X, np.zeros_like(X))), sequence_length, y)
    log_message += '\n' + str(datetime.now()) + '\tTest set result\t'
    log_message += score_template % D.evaluate(np.hstack((X_test, np.zeros_like(X_test))), test_sequence_length, y_test)
    with open(log_path, 'a') as f:
        f.write(log_message + '\n')

    for i in range(50):
        log_message = str(datetime.now()) + '\tTraining generative model for the %d-th time\n' % (i,)
        #G.train(training_data_malware[:, :-1])
        G.train((X_malware, X_benign), (malware_length, benign_length))
        log_message += str(datetime.now()) + '\tGenerating examples\n'
        generated_training_malware, generated_training_malware_length = \
            G.sample((X_malware, X_benign), (malware_length, benign_length))
        generated_training_data = \
            np.vstack((generated_training_malware, np.hstack((X_benign, np.zeros_like(X_benign)))))
        generated_sequence_length = np.hstack((generated_training_malware_length, benign_length))
        generated_test_malware, generated_test_malware_length = \
            G.sample((X_malware_test, X_benign), (malware_length_test, benign_length))
        generated_test_data = \
            np.vstack((generated_test_malware, np.hstack((X_benign_test, np.zeros_like(X_benign_test)))))
        generated_sequence_length_test = np.hstack((generated_test_malware_length, benign_length_test))

        log_message += str(datetime.now()) + '\tTraining set result before re-training\t'
        log_message += score_template % D.evaluate(generated_training_data, generated_sequence_length, y)
        log_message += '\n' + str(datetime.now()) + '\tTest set result before re-training\t'
        log_message += score_template % D.evaluate(generated_test_data, generated_sequence_length_test, y_test)
        log_message += '\n' + str(datetime.now()) + '\tRe-training discriminative model\n'
        D.train(generated_training_data, generated_sequence_length, y)
        log_message += str(datetime.now()) + '\tTraining set result after re-training\t'
        log_message += score_template % D.evaluate(generated_training_data, generated_sequence_length, y)
        log_message += '\n' + str(datetime.now()) + '\tTest set result after re-training\t'
        log_message += score_template % D.evaluate(generated_test_data, generated_sequence_length_test, y_test)
        with open(log_path, 'a') as f:
            f.write(log_message + '\n\n')

def learning_SeqInserter():
    cell_type = 'LSTM'
    G_layers = [128]
    G_bidirectional = True
    D_layers = [128]
    D_attention_layers = [128]
    D_ff_layers = [128]
    batch_size = 128
    benign_batch_size = 53
    max_length = 1024
    max_epoch = 1000
    max_epoch_no_improvement = 5
    learning_rate = 0.001
    temperature = 60.0
    regularization = 0.0001

    tag = '20171105_rnn_biLSTM_%s_G_%s_%s_D_%s_attention_%s_ff_%s_max_len_%d_lr_%g_temp_%g_regu_%g' % (
        cell_type,
        '_'.join([str(layer) for layer in G_layers]),
        'birnn' if G_bidirectional else 'rnn',
        '_'.join([str(layer) for layer in D_layers]),
        '_'.join([str(layer) for layer in D_attention_layers]),
        '_'.join([str(layer) for layer in D_ff_layers]),
        max_length,
        learning_rate,
        temperature,
        regularization
    )
    dir_path = '../seq_models/' + tag
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    if os.path.exists(os.path.join(dir_path, 'code')):
        shutil.rmtree(os.path.join(dir_path, 'code'))
    shutil.copytree('.', os.path.join(dir_path, 'code'))

    model_path = dir_path + '/model'
    log_path = dir_path + '/log.txt'
    score_template = 'TPR %(TPR)f\tFPR %(FPR)f\tAccuracy %(Accuracy)f\tAUC %(AUC)f'
    D = _get_rnn_model(dir_path + '/seq_discriminator_model', max_length=2 * max_length)
    G = SeqInserter(D, cell_type=cell_type, G_layers=G_layers, G_bidirectional=G_bidirectional,
                    D_layers=D_layers, D_attention_layers=D_attention_layers, D_ff_layers=D_ff_layers,
                    batch_size=batch_size, benign_batch_size=benign_batch_size, num_tokens=161,
                    max_length=max_length, max_epoch=max_epoch, max_epoch_no_improvement=max_epoch_no_improvement,
                    learning_rate=learning_rate, temperature=temperature, regularization=regularization,
                    model_path=model_path)

    # load the data
    X_malware, malware_length, X_benign, benign_length = \
        load_dataset('../data/API_rand_trainval_len_2048.txt', max_length, 0)
    X = np.vstack((X_malware, X_benign))
    sequence_length = np.hstack((malware_length, benign_length))
    y = np.array([1] * len(X_malware) + [0] * len(X_benign))
    X_malware_test, malware_length_test, X_benign_test, benign_length_test = \
        load_dataset('../data/API_rand_test_len_2048.txt', max_length, 0)
    X_test = np.vstack((X_malware_test, X_benign_test))
    test_sequence_length = np.hstack((malware_length_test, benign_length_test))
    y_test = np.array([1] * len(X_malware_test) + [0] * len(X_benign_test))

    log_message = str(datetime.now()) + '\tTraining discriminative model on original dataset\n'
    D.train(np.hstack((X, np.zeros_like(X))), sequence_length, y)
    #log_message += str(datetime.now()) + '\tTraining set result\t'
    #log_message += score_template % D.evaluate(np.hstack((X, np.zeros_like(X))), sequence_length, y)
    #log_message += '\n' + str(datetime.now()) + '\tTest set result\t'
    #log_message += score_template % D.evaluate(np.hstack((X_test, np.zeros_like(X_test))), test_sequence_length, y_test)
    with open(log_path, 'a') as f:
        f.write(log_message + '\n')

    for i in range(50):
        log_message = str(datetime.now()) + '\tTraining generative model for the %d-th time\n' % (i,)
        #G.train(training_data_malware[:, :-1])
        G.train((X_malware, X_benign), (malware_length, benign_length))
        log_message += str(datetime.now()) + '\tGenerating examples\n'
        generated_training_malware, generated_training_malware_length = \
            G.sample((X_malware, X_benign), (malware_length, benign_length))
        generated_training_data = \
            np.vstack((generated_training_malware, np.hstack((X_benign, np.zeros_like(X_benign)))))
        generated_sequence_length = np.hstack((generated_training_malware_length, benign_length))
        generated_test_malware, generated_test_malware_length = \
            G.sample((X_malware_test, X_benign), (malware_length_test, benign_length))
        generated_test_data = \
            np.vstack((generated_test_malware, np.hstack((X_benign_test, np.zeros_like(X_benign_test)))))
        generated_sequence_length_test = np.hstack((generated_test_malware_length, benign_length_test))

        log_message += str(datetime.now()) + '\tTraining set result before re-training\t'
        log_message += score_template % D.evaluate(generated_training_data, generated_sequence_length, y)
        log_message += '\n' + str(datetime.now()) + '\tTest set result before re-training\t'
        log_message += score_template % D.evaluate(generated_test_data, generated_sequence_length_test, y_test)
        log_message += '\n' + str(datetime.now()) + '\tRe-training discriminative model\n'
        D.train(generated_training_data, generated_sequence_length, y)
        log_message += str(datetime.now()) + '\tTraining set result after re-training\t'
        log_message += score_template % D.evaluate(generated_training_data, generated_sequence_length, y)
        log_message += '\n' + str(datetime.now()) + '\tTest set result after re-training\t'
        log_message += score_template % D.evaluate(generated_test_data, generated_sequence_length_test, y_test)
        with open(log_path, 'a') as f:
            f.write(log_message + '\n\n')

if __name__ == '__main__':
    #tune_discriminator_parameters()
    learning_SeqInserter()
