import numpy as np
import tensorflow as tf
import math

from .base_model import BaseModel
from .data_utils import minibatches, pad_sequences, get_chunks, judge_ooxv, write_prediction
from .general_utils import Progbar

NUM = "$NUM$"


class NERModel(BaseModel):
    """Specialized class of Model for NER"""

    def __init__(self, config):
        super(NERModel, self).__init__(config)
        self.idx_to_tag = {idx: tag for tag, idx in
                           self.config.vocab_tags.items()}

    def add_placeholders(self):
        """Define placeholders = entries to computational graph"""
        # shape = (batch size, max length of sentence in batch)
        self.word_ids = tf.placeholder(tf.int32, shape=[None, None],
                                       name="word_ids")

        # shape = (batch size)
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None],
                                               name="sequence_lengths")

        # shape = (batch size, max length of sentence, max length of word)
        self.char_ids = tf.placeholder(tf.int32, shape=[None, None, None],
                                       name="char_ids")

        # shape = (batch_size, max_length of sentence)
        self.word_lengths = tf.placeholder(tf.int32, shape=[None, None],
                                           name="word_lengths")

        # shape = (batch size, max length of sentence in batch)
        self.labels = tf.placeholder(tf.int32, shape=[None, None],
                                     name="labels")

        # hyper parameters
        self.dropout = tf.placeholder(dtype=tf.float32, shape=[],
                                      name="dropout")
        self.lr = tf.placeholder(dtype=tf.float32, shape=[],
                                 name="lr")

    def get_feed_dict(self, words, labels=None, lr=None, dropout=None):
        """Given some data, pad it and build a feed dictionary

        Args:
            words: list of sentences. A sentence is a list of ids of a list of
                words. A word is a list of ids
            labels: list of ids
            lr: (float) learning rate
            dropout: (float) keep prob

        Returns:
            dict {placeholder: value}

        """
        # perform padding of the given data
        if self.config.use_chars:
            char_ids, word_ids = zip(*words)
            word_ids, sequence_lengths = pad_sequences(word_ids, 0)
            char_ids, word_lengths = pad_sequences(char_ids, pad_tok=0,
                                                   nlevels=2)
        else:
            word_ids, sequence_lengths = pad_sequences(words, 0)

        # build feed dictionary
        feed = {
            self.word_ids: word_ids,
            self.sequence_lengths: sequence_lengths
        }

        if self.config.use_chars:
            feed[self.char_ids] = char_ids
            feed[self.word_lengths] = word_lengths

        if labels is not None:
            labels, _ = pad_sequences(labels, 0)
            feed[self.labels] = labels

        if lr is not None:
            feed[self.lr] = lr

        if dropout is not None:
            feed[self.dropout] = dropout

        return feed, sequence_lengths

    def add_word_embeddings_op(self):
        """Defines self.word_embeddings

        If self.config.embeddings is not None and is a np array initialized
        with pre-trained word vectors, the word embeddings is just a look-up
        and we don't train the vectors. Otherwise, a random matrix with
        the correct shape is initialized.
        """
        with tf.variable_scope("words"):
            if self.config.embeddings is None:
                self.logger.info("WARNING: randomly initializing word vectors")
                _word_embeddings = tf.get_variable(
                    name="_word_embeddings",
                    dtype=tf.float32,
                    shape=[self.config.nwords, self.config.dim_word])
                _word_embeddings_proj = _word_embeddings
            else:
                _word_embeddings = tf.Variable(
                    self.config.embeddings,
                    name="_word_embeddings",
                    dtype=tf.float32,
                    trainable=self.config.train_embeddings)
                _word_embeddings_proj = _word_embeddings

                if self.config.use_projection:
                    if self.config.embedding_projection_type == "linear":
                        if self.config.projection_w_initilization == "xavier":
                            w = tf.get_variable(
                                name="W_embedding",
                                shape=[self.config.dim_word, self.config.dim_word],
                                initializer=tf.truncated_normal_initializer(
                                    stddev=math.sqrt(2 / self.config.dim_word)),
                                dtype=tf.float32,
                                trainable=True)
                            self.regularizer = tf.contrib.layers.l2_regularizer(1.0 / self.config.dim_word)(w)
                            _word_embeddings_proj = tf.matmul(_word_embeddings, w)
                            _word_embeddings_proj = tf.contrib.layers.batch_norm(_word_embeddings_proj, center=True,
                                                                                 scale=True,
                                                                                 is_training=True)
                        elif self.config.projection_w_initilization == "eye":
                            w = tf.get_variable(
                                name="W_embedding",
                                initializer=tf.eye(self.config.dim_word),
                                dtype=tf.float32,
                                trainable=True)
                            b = tf.get_variable(
                                name="b_embedding",
                                shape=[1, self.config.dim_word],
                                initializer=tf.zeros_initializer,
                                dtype=tf.float32,
                                trainable=True)
                            self.regularizer = tf.contrib.layers.l2_regularizer(1.0 / self.config.dim_word)(w)
                            _word_embeddings_proj = tf.matmul(_word_embeddings, w) + tf.tile(b, [self.config.nwords, 1])

                    elif self.config.embedding_projection_type == "relu":
                        w1 = tf.get_variable(
                            name="W1_embedding",
                            shape=[self.config.dim_word, self.config.dim_word],
                            initializer=tf.truncated_normal_initializer(
                                stddev=math.sqrt(2 / self.config.dim_word)),
                            dtype=tf.float32,
                            trainable=True)
                        out = tf.matmul(_word_embeddings, w1)
                        out = tf.contrib.layers.batch_norm(out, center=True, scale=True, is_training=True)
                        out = tf.nn.relu(out)
                        w2 = tf.get_variable(
                            name="W2_embedding",
                            shape=[self.config.dim_word, self.config.dim_word],
                            initializer=tf.truncated_normal_initializer(
                                stddev=math.sqrt(2 / self.config.dim_word)),
                            dtype=tf.float32,
                            trainable=True)
                        self.regularizer = tf.contrib.layers.l2_regularizer(1.0 / self.config.dim_word)(
                            w1) + tf.contrib.layers.l2_regularizer(1.0 / self.config.dim_word)(w2)
                        _word_embeddings_proj = tf.nn.relu(tf.contrib.layers.batch_norm(tf.matmul(out, w2)))

                    if self.config.use_residual:
                        if self.config.use_attention:
                            s = tf.get_variable(
                                name="s_attention",
                                shape=[self.config.dim_word, 1],
                                dtype=tf.float32,
                                trainable=True)
                            alpha = tf.matrix_diag(tf.transpose(tf.tanh(tf.matmul(_word_embeddings, s))))[0]
                            _word_embeddings_proj = tf.matmul(alpha, _word_embeddings_proj)
                        _word_embeddings_proj += _word_embeddings

                if self.config.copy_embeddings:
                    _word_embeddings_temp = tf.Variable(
                        self.config.embeddings,
                        name="_word_embeddings_temp",
                        dtype=tf.float32,
                        trainable=False)
                    _word_embeddings_proj = tf.concat([_word_embeddings_proj, _word_embeddings_temp], axis=-1)

            self.word_embeddings_proj = _word_embeddings_proj
            word_embeddings = tf.nn.embedding_lookup(_word_embeddings_proj,
                                                     self.word_ids, name="word_embeddings")

        with tf.variable_scope("chars"):
            if self.config.use_chars:
                # get char embeddings matrix
                _char_embeddings = tf.get_variable(
                    name="_char_embeddings",
                    dtype=tf.float32,
                    shape=[self.config.nchars, self.config.dim_char])
                char_embeddings = tf.nn.embedding_lookup(_char_embeddings,
                                                         self.char_ids, name="char_embeddings")

                # put the time dimension on axis=1
                s = tf.shape(char_embeddings)
                char_embeddings = tf.reshape(char_embeddings,
                                             shape=[s[0] * s[1], s[-2], self.config.dim_char])
                word_lengths = tf.reshape(self.word_lengths, shape=[s[0] * s[1]])

                # bi lstm on chars
                cell_fw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_char,
                                                  state_is_tuple=True)
                cell_bw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_char,
                                                  state_is_tuple=True)
                _output = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw, cell_bw, char_embeddings,
                    sequence_length=word_lengths, dtype=tf.float32)

                # read and concat output
                _, ((_, output_fw), (_, output_bw)) = _output
                output = tf.concat([output_fw, output_bw], axis=-1)

                # shape = (batch size, max sentence length, char hidden size)
                output = tf.reshape(output,
                                    shape=[s[0], s[1], 2 * self.config.hidden_size_char])
                word_embeddings = tf.concat([word_embeddings, output], axis=-1)

        self.word_embeddings = tf.nn.dropout(word_embeddings, self.dropout)

    def add_logits_op(self):
        """Defines self.logits

        For each word in each sentence of the batch, it corresponds to a vector
        of scores, of dimension equal to the number of tags.
        """
        with tf.variable_scope("bi-lstm"):
            cell_fw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_lstm)
            cell_bw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_lstm)
            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw, cell_bw, self.word_embeddings,
                sequence_length=self.sequence_lengths, dtype=tf.float32)
            output = tf.concat([output_fw, output_bw], axis=-1)
            output = tf.nn.dropout(output, self.dropout)

        with tf.variable_scope("proj"):
            # Notice! There is a rand number.
            W = tf.get_variable("W", dtype=tf.float32,
                                shape=[2 * self.config.hidden_size_lstm, self.config.ntags])

            b = tf.get_variable("b", shape=[self.config.ntags],
                                dtype=tf.float32, initializer=tf.zeros_initializer())

            nsteps = tf.shape(output)[1]
            output = tf.reshape(output, [-1, 2 * self.config.hidden_size_lstm])
            pred = tf.matmul(output, W) + b
            self.logits = tf.reshape(pred, [-1, nsteps, self.config.ntags])

    def add_pred_op(self):
        """Defines self.labels_pred

        This op is defined only in the case where we don't use a CRF since in
        that case we can make the prediction "in the graph" (thanks to tf
        functions in other words). With theCRF, as the inference is coded
        in python and not in pure tensroflow, we have to make the prediciton
        outside the graph.
        """
        if not self.config.use_crf:
            self.labels_pred = tf.cast(tf.argmax(self.logits, axis=-1),
                                       tf.int32)

    def add_loss_op(self):
        """Defines the loss"""
        if self.config.use_crf:
            log_likelihood, trans_params = tf.contrib.crf.crf_log_likelihood(
                self.logits, self.labels, self.sequence_lengths)
            self.trans_params = trans_params  # need to evaluate it for decoding
            self.loss = tf.reduce_mean(-log_likelihood)
        else:
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.logits, labels=self.labels)
            mask = tf.sequence_mask(self.sequence_lengths)
            losses = tf.boolean_mask(losses, mask)
            self.loss = tf.reduce_mean(losses)
            if self.config.use_projection and self.config.use_projection_regularizer:
                self.loss += self.regularizer

        # for tensorboard
        tf.summary.scalar("loss", self.loss)

    def build(self):
        # NER specific functions
        self.add_placeholders()
        self.add_word_embeddings_op()
        self.add_logits_op()
        self.add_pred_op()
        self.add_loss_op()

        # Generic functions that add training op and initialize session
        self.add_train_op(self.config.lr_method, self.lr, self.loss,
                          self.config.clip)
        self.initialize_session()  # now self.sess is defined and vars are init

    def predict_batch(self, words):
        """
        Args:
            words: list of sentences

        Returns:
            labels_pred: list of labels for each sentence
            sequence_length

        """
        fd, sequence_lengths = self.get_feed_dict(words, dropout=1.0)

        if self.config.use_crf:
            # get tag scores and transition params of CRF
            viterbi_sequences = []
            logits, trans_params = self.sess.run(
                [self.logits, self.trans_params], feed_dict=fd)

            # iterate over the sentences because no batching in vitervi_decode
            for logit, sequence_length in zip(logits, sequence_lengths):
                logit = logit[:sequence_length]  # keep only the valid steps
                viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(
                    logit, trans_params)
                viterbi_sequences += [viterbi_seq]

            return viterbi_sequences, sequence_lengths

        else:
            labels_pred = self.sess.run(self.labels_pred, feed_dict=fd)

            return labels_pred, sequence_lengths

    def run_epoch(self, train, dev, epoch):
        """Performs one complete pass over the train set and evaluate on dev

        Args:
            train: dataset that yields tuple of sentences, tags
            dev: dataset
            epoch: (int) index of the current epoch

        Returns:
            f1: (python float), score to select model on, higher is better

        """
        # progbar stuff for logging
        batch_size = self.config.batch_size
        nbatches = (len(train) + batch_size - 1) // batch_size
        prog = Progbar(target=nbatches)

        # iterate over dataset
        for i, (words, labels, raw_words) in enumerate(minibatches(train, batch_size)):
            fd, _ = self.get_feed_dict(words, labels, self.config.lr,
                                       self.config.dropout)

            _, train_loss, summary = self.sess.run(
                [self.train_op, self.loss, self.merged], feed_dict=fd)

            prog.update(i + 1, [("train loss", train_loss)])

            # tensorboard
            if i % 10 == 0:
                self.file_writer.add_summary(summary, epoch * nbatches + i)

        metrics, metrics2 = self.run_evaluate(dev)
        msg = " - ".join(["{} {:04.2f}".format(k, v)
                          for k, v in metrics.items()])
        msg2 = " - ".join(["{} {:04.2f}".format(k, v)
                           for k, v in metrics2.items()])
        self.logger.info(msg + "\n" + msg2)

        return metrics["f1"]

    def run_evaluate(self, test, write_mistake_2file=False):
        """Evaluates performance on test set

        Args:
            test: dataset that yields tuple of (sentences, tags)

        Returns:
            metrics: (dict) metrics["acc"] = 98.4, ...

        """
        if write_mistake_2file:
            # clear the file
            fin = open(self.config.filename_wrong_preds, "w", encoding="utf-8")
            fin.close()
        UNK = "$UNK$"
        accs = []
        correct_preds, total_correct, total_preds = 0., 0., 0.
        correct_preds_ooxv, total_correct_ooxv, total_preds_ooxv = [0.] * 4, [0.] * 4, [0.] * 4
        for words, labels, raw_words in minibatches(test, self.config.batch_size):
            labels_pred, sequence_lengths = self.predict_batch(words)

            for sen, lab, lab_pred, length in zip(raw_words, labels, labels_pred,
                                                  sequence_lengths):
                sen = sen[:length]
                lab = lab[:length]
                lab_pred = lab_pred[:length]
                accs += [a == b for (a, b) in zip(lab, lab_pred)]

                lab_chunks = set(get_chunks(lab, self.config.vocab_tags))
                lab_pred_chunks = set(get_chunks(lab_pred,
                                                 self.config.vocab_tags))

                correct_preds += len(lab_chunks & lab_pred_chunks)
                total_preds += len(lab_pred_chunks)
                total_correct += len(lab_chunks)

                if write_mistake_2file:
                    write_prediction(self.config.filename_wrong_preds, sen, lab_chunks, lab_pred_chunks)

                for chunk in lab_chunks:
                    total_correct_ooxv[judge_ooxv(sen, chunk, self.config.lowercase, self.config.vocab_trainset_word,
                                                  self.config.vocab_embedding_word)] += 1
                for chunk in lab_pred_chunks:
                    total_preds_ooxv[judge_ooxv(sen, chunk, self.config.lowercase, self.config.vocab_trainset_word,
                                                self.config.vocab_embedding_word)] += 1
                for chunk in (lab_pred_chunks & lab_chunks):
                    correct_preds_ooxv[judge_ooxv(sen, chunk, self.config.lowercase, self.config.vocab_trainset_word,
                                                  self.config.vocab_embedding_word)] += 1

        pp = correct_preds / total_preds if correct_preds > 0 else 0
        rr = correct_preds / total_correct if correct_preds > 0 else 0
        ff1 = 2 * pp * rr / (pp + rr) if correct_preds > 0 else 0
        p, r, f1 = [0] * 4, [0] * 4, [0] * 4
        for i in range(4):
            p[i] = correct_preds_ooxv[i] / total_preds_ooxv[i] if correct_preds_ooxv[i] > 0 else 0
            r[i] = correct_preds_ooxv[i] / total_correct_ooxv[i] if correct_preds_ooxv[i] > 0 else 0
            f1[i] = 2 * p[i] * r[i] / (p[i] + r[i]) if correct_preds_ooxv[i] > 0 else 0

        acc = np.mean(accs)

        return {"acc": 100 * acc, "f1": 100 * ff1, "num": total_correct}, \
               {"oobv_num": total_correct_ooxv[0], "oobv_f1": f1[0] * 100,
                "ooev_num": total_correct_ooxv[1], "ooev_f1": f1[1] * 100,
                "ootv_num": total_correct_ooxv[2], "ootv_f1": f1[2] * 100,
                "iv_num": total_correct_ooxv[3], "iv_f1": f1[3] * 100}

    def predict(self, words_raw):
        """Returns list of tags

        Args:
            words_raw: list of words (string), just one sentence (no batch)

        Returns:
            preds: list of tags (string), one for each word in the sentence

        """
        words = [self.config.processing_word(w) for w in words_raw]
        if type(words[0]) == tuple:
            words = zip(*words)
        pred_ids, _ = self.predict_batch([words])
        preds = [self.idx_to_tag[idx] for idx in list(pred_ids[0])]

        return preds

    def get_embeddings(self, test):
        for words, _, _ in minibatches(test, self.config.batch_size):
            fd, sequence_lengths = self.get_feed_dict(words, dropout=1.0)
            word_embeddings = self.sess.run(self.word_embeddings_proj, feed_dict=fd)
            return word_embeddings
