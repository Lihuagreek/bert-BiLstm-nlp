# -*- coding:utf-8 -*-
import argparse
import json
import math
import pickle as cPickle
import random
import sys

import numpy as np
import tensorflow as tf
from bert_base.bert import modeling, optimization, tokenization
from bert_base.bert.optimization import create_optimizer
from bert_base.train import tf_metrics
from tensorflow.contrib import rnn
from tensorflow.contrib.crf import crf_log_likelihood, viterbi_decode
from tensorflow.contrib.layers.python.layers import initializers

from helper import ARGS
from utils import f1_score, format_result, get_tags, new_f1_score


class Model():
    def __init__(self):
        self.nums_tags = 4
        self.lstm_dim = 128
        self.embedding_size = 50
        self.max_epoch = 10
        self.learning_rate = ARGS.learning_rate
        self.global_steps = tf.Variable(0, trainable=False)
        # tf.Variable变量定义，变量需要再session中初始化和运行
        self.best_dev_f1 = tf.Variable(0.0, trainable=False)
        self.checkpoint_dir = "./model/"
        self.checkpoint_path = "./model/ner.org.ckpt"
        self.initializer = initializers.xavier_initializer()

        self.is_training = True if ARGS.entry == "train" else False

    def __create_model(self):

        # embbeding layer
        if ARGS.mode == "bert":
            self._init_bert_placeholder()
            self.bert_layer()
        else:
            self._init_placeholder()
            self.embedding_layer()

        # bi-Lstm layer
        self.biLSTM_layer()

        # logits_layer
        self.logits_layer()

        # loss_layer
        self.loss_layer()

        # optimizer_layer
        if ARGS.mode == "bert":
            self.bert_optimizer_layer()
        else:
            self.optimizer_layer()

    def _init_placeholder(self):

        self.inputs = tf.placeholder(
            dtype=tf.int32,
            shape=[None, None],
            name="Inputs"
        )

        self.targets = tf.placeholder(
            dtype=tf.int32,
            shape=[None, None],
            name="Targets"
        )

        self.dropout = tf.placeholder(
            dtype=tf.float32,
            shape=None,
            name="Dropout"
        )

        used = tf.sign(tf.abs(self.inputs))
        length = tf.reduce_sum(used, reduction_indices=1)
        self.length = tf.cast(length, tf.int32)
        self.batch_size = tf.shape(self.inputs)[0]
        self.nums_steps = tf.shape(self.inputs)[-1]

    def _init_bert_placeholder(self):
        self.input_ids = tf.placeholder(
            dtype=tf.int32,
            shape=[None, None],
            name="bert_input_ids"
        )
        self.input_mask = tf.placeholder(
            dtype=tf.int32,
            shape=[None, None],
            name="bert_input_mask"
        )
        self.segment_ids = tf.placeholder(
            dtype=tf.int32,
            shape=[None, None],
            name="bert_segment_ids"
        )
        self.targets = tf.placeholder(
            dtype=tf.int32,
            shape=[None, None],
            name="bert_targets"
        )

        self.dropout = tf.placeholder(
            dtype=tf.float32,
            shape=None,
            name="bert_dropout"
        )
        # tf.sign(x),x<0,0;x=0,0;x>0,1
        used = tf.sign(tf.abs(self.input_ids))
        # 维度减1 列相加
        length = tf.reduce_sum(used, reduction_indices=1)
        self.length = tf.cast(length, tf.int32)
        # self.batch_size = tf.shape(self.input_ids)[0]
        self.nums_steps = tf.shape(self.input_ids)[-1]

    def bert_layer(self):
        bert_config = modeling.BertConfig.from_json_file(ARGS.bert_config)

        model = modeling.BertModel(
            config=bert_config,
            is_training=self.is_training,
            input_ids=self.input_ids,
            input_mask=self.input_mask,
            token_type_ids=self.segment_ids,
            use_one_hot_embeddings=False
        )
        self.embedded = model.get_sequence_output()
        self.model_inputs = tf.nn.dropout(
            self.embedded, self.dropout
        )

    def embedding_layer(self):
        with tf.variable_scope("embedding_layer") as scope:
            sqart3 = math.sqrt(3)

            self.embedding_matrix = tf.get_variable(
                name="embedding_matrix",
                shape=[self.input_size, self.embedding_size],
                initializer=self.initializer,
                dtype=tf.float32,
            )

            self.embedded = tf.nn.embedding_lookup(
                self.embedding_matrix, self.inputs
            )

            self.model_inputs = tf.nn.dropout(
                self.embedded, self.dropout
            )

    def biLSTM_layer(self):
        with tf.variable_scope("bi-LSTM") as scope:
            lstm_cell = {}
            for direction in ["forward", "backward"]:
                with tf.variable_scope(direction):
                    lstm_cell[direction] = rnn.GRUCell(
                        num_units=self.lstm_dim,
                        # use_peepholes=True,
                        # initializer=self.initializer,
                        # state_is_tuple=True
                    )

            outputs, final_states = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=lstm_cell['forward'],
                cell_bw=lstm_cell['backward'],
                inputs=self.model_inputs,
                sequence_length=self.length,
                dtype=tf.float32,
            )
            self.lstm_outputs = tf.concat(outputs, axis=2)

    def logits_layer(self):
        with tf.variable_scope("hidden"):
            w = tf.get_variable("W", shape=[self.lstm_dim * 2, self.lstm_dim],
                                dtype=tf.float32, initializer=self.initializer
                                )
            b = tf.get_variable("b", shape=[self.lstm_dim], dtype=tf.float32,
                                initializer=self.initializer
                                )

            output = tf.reshape(self.lstm_outputs, shape=[-1, self.lstm_dim * 2])
            hidden = tf.tanh(tf.nn.xw_plus_b(output, w, b))
            self.hidden = hidden

        with tf.variable_scope("logits"):
            w = tf.get_variable("W", shape=[self.lstm_dim, self.nums_tags],
                                initializer=self.initializer, dtype=tf.float32
                                )
            self.test_w = w
            b = tf.get_variable("b", shape=[self.nums_tags], dtype=tf.float32)
            self.test_b = b
            pred = tf.nn.xw_plus_b(hidden, w, b)
            self.logits = tf.reshape(
                pred, shape=[-1, self.nums_steps, self.nums_tags])

    def loss_layer(self):
        with tf.variable_scope("loss_layer"):
            logits = self.logits
            targets = self.targets

            self.trans = tf.get_variable(
                "transitions",
                shape=[self.nums_tags, self.nums_tags],
                initializer=self.initializer
            )
            # inputs预测值；tag_indices真实值；sequence_lengths样本真实的序列长度；transition_params转移概率
            # log_likelihood是预测值和真实值尽可能的相近时的参数值
            log_likelihood, self.trans = crf_log_likelihood(
                inputs=logits,
                tag_indices=targets,
                transition_params=self.trans,
                sequence_lengths=self.length
            )
            self.loss = tf.reduce_mean(-log_likelihood)

    def bert_optimizer_layer(self):
        # tf.argmax(self.logits, 2)返回每行最大值的索引，tf.equal对比预测标签和真实标签索引是否相等
        correct_prediction = tf.equal(tf.argmax(self.logits, 2), tf.cast(self.targets, tf.int64))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        num_train_steps = int(self.train_length / self.batch_size * self.max_epoch)
        num_warmup_steps = int(num_train_steps * 0.1)
        self.train_op = create_optimizer(self.loss, self.learning_rate, num_train_steps, num_warmup_steps, False)
        # tf.global_variables()查看全部变量；max_to_keep保存最近的5个模型
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=7)

    def optimizer_layer(self):
        with tf.variable_scope("optimizer"):
            optimizer = tf.train.AdamOptimizer()

            correct_prediction = tf.equal(
                tf.argmax(self.logits, 2), tf.cast(self.targets, tf.int64))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            # 返回可训练的变量
            tvars = tf.trainable_variables()
            # loss对tvars求导
            grads = tf.gradients(self.loss, tvars)

            # This is how the model was pre-trained.预训练模型
            (grads, _) = tf.clip_by_global_norm(grads, clip_norm=1.0)

            self.train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_steps)
            self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=7)

    def step(self, sess, batch):

        # zip(*) 可理解为解压，返回二维矩阵式
        # zip() 将元素组成元组返回
        inputs, targets = zip(*batch)

        feed = {
            self.inputs: inputs,
            self.targets: targets,
            self.dropout: 0.5
        }
        embedding, global_steps, loss, _, logits, acc, length = sess.run(
            [self.embedded, self.global_steps, self.loss, self.train_op, self.logits, self.accuracy, self.length],
            feed_dict=feed)
        return global_steps, loss, logits, acc, length

    def bert_step(self, sess, batch):
        ntokens, tag_ids, inputs_ids, segment_ids, input_mask = zip(*batch)

        # feed设置graph输入值
        feed = {
            self.input_ids: inputs_ids,
            self.targets: tag_ids,
            self.segment_ids: segment_ids,
            self.input_mask: input_mask,
            self.dropout: 0.5
        }
        embedding, global_steps, loss, _, logits, acc, length = sess.run(
            [self.embedded, self.global_steps, self.loss, self.train_op, self.logits, self.accuracy, self.length],
            feed_dict=feed)
        return global_steps, loss, logits, acc, length

    def train(self):
        """
        train: 训练集是用来训练模型的
        dev: 用来对训练集训练出来的模型进行测试，优化模型
        test: 对训练出的模型进行一次最终的评估所用的数据集
        :return:
        """
        # bert模式
        if ARGS.mode == "bert":
            from bert_data_utils import BertDataUtils
            tokenizer = tokenization.FullTokenizer(
                vocab_file=ARGS.vocab_dir,
            )
            self.train_data = BertDataUtils(tokenizer, batch_size=10)
            self.dev_data = BertDataUtils(tokenizer, batch_size=300)
            self.dev_batch = self.dev_data.iteration()
        # 一般模型
        else:
            from data_utils import DataBatch
            self.train_data = DataBatch(data_type='train', batch_size=100)

            self.vocab = self.train_data.vocab
            self.input_size = len(self.vocab.values()) + 1
            self.dev_data = DataBatch(data_type='dev', batch_size=300)
            self.dev_batch = self.dev_data.iteration()

        data = {
            "batch_size": self.train_data.batch_size,
            "input_size": self.train_data.input_size,
            "vocab": self.train_data.vocab,
            "tag_map": self.train_data.tag_map,
        }

        f = open("data/data_map.pkl", "wb")
        # cPickle.dump(data, f)按照data字典定义的格式，保存到文件流f中
        cPickle.dump(data, f)
        f.close()
        self.batch_size = self.train_data.batch_size
        self.nums_tags = len(self.train_data.tag_map.keys())
        self.tag_map = self.train_data.tag_map
        self.train_length = len(self.train_data.data)

        # self.test_data = DataBatch(data_type='test', batch_size=100)
        # self.test_batch = self.test_data.get_batch()
        # save vocab
        print("-" * 50)
        print("train data:\t", self.train_length)
        print("nums of tags:\t", self.nums_tags)
        # 创建模型
        self.__create_model()
        with tf.Session() as sess:
            with tf.device("/gpu:0"):
                # tf.train.get_checkpoint_state函数通过checkpoint文件找到模型文件名。
                ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
                # ckpt.model_checkpoint_path保存了最新的tensorflow模型文件的文件名
                if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
                    print("restore model")
                    # 提取训练好的模型的变量参数
                    self.saver.restore(sess, ckpt.model_checkpoint_path)
                else:
                    sess.run(tf.global_variables_initializer())

                tvars = tf.trainable_variables()
                (assignment_map, initialized_variable_names) = \
                    modeling.get_assignment_map_from_checkpoint(tvars,
                                                                ARGS.init_checkpoint)
                # 利用给定的模型的tenser初始化变量
                tf.train.init_from_checkpoint(ARGS.init_checkpoint, assignment_map)
                for var in tvars:
                    init_string = ""
                    if var.name in initialized_variable_names:
                        init_string = ", *INIT_FROM_CKPT*"
                    print("  name = %s, shape = %s%s", var.name, var.shape,
                          init_string)
                for i in range(self.max_epoch):
                    print("-" * 50)
                    print("epoch {}".format(i))

                    steps = 0
                    for batch in self.train_data.get_batch():
                        steps += 1
                        if ARGS.mode == "bert":
                            global_steps, loss, logits, acc, length = self.bert_step(
                                sess, batch
                            )
                        else:
                            global_steps, loss, logits, acc, length = self.step(
                                sess, batch
                            )
                        if steps % 1 == 0:
                            print("[->] step {}/{}\tloss {:.2f}\tacc {:.2f}".format(
                                steps, len(self.train_data.batch_data), loss, acc))
                    if ARGS.mode == "bert":
                        self.bert_evaluate(sess, "ORG")
                        self.bert_evaluate(sess, "LOC")
                    else:
                        self.evaluate(sess, "ORG")
                        self.evaluate(sess, "LOC")
                    self.saver.save(sess, self.checkpoint_path)

    def decode(self, scores, lengths, trans):
        paths = []
        for score, length in zip(scores, lengths):
            path, _ = viterbi_decode(score, trans)
            paths.append(path)
        return paths

    def bert_evaluate(self, sess, tag):
        """
        recall: 为正的样本中被预测为正样本的概率
        precision: 在被所有预测为正的样本中实际为正样本的概率
        f1:
        :param sess:
        :param tag:
        :return:
        """
        result = []
        trans = self.trans.eval()
        batch = self.dev_batch.__next__()

        ntokens, tag_ids, inputs_ids, segment_ids, input_mask = zip(*batch)
        feed = {
            self.input_ids: inputs_ids,
            self.segment_ids: segment_ids,
            self.targets: tag_ids,
            self.input_mask: input_mask,
            self.dropout: 1
        }
        scores, acc, lengths = sess.run(
            [self.logits, self.accuracy, self.length], feed_dict=feed)
        pre_paths = self.decode(scores, lengths, trans)

        tar_paths = tag_ids
        recall, precision, f1 = f1_score(
            tar_paths, pre_paths, tag, self.tag_map)
        # recall, precision, f1 = new_f1_score(
        #     tar_paths, pre_paths, tag, self.tag_map)
        best = self.best_dev_f1.eval()
        if f1 > best:
            print("\tnew best f1:")
            print("\trecall {:.2f}\t precision {:.2f}\t f1 {:.2f}".format(
                recall, precision, f1))
            tf.assign(self.best_dev_f1, f1).eval()

    def evaluate(self, sess, tag):
        result = []
        trans = self.trans.eval()
        batch = self.dev_batch.__next__()
        inputs, targets = zip(*batch)
        feed = {
            self.inputs: inputs,
            self.targets: targets,
            self.dropout: 1
        }
        scores, acc, lengths = sess.run(
            [self.logits, self.accuracy, self.length], feed_dict=feed)

        pre_paths = self.decode(scores, lengths, trans)

        tar_paths = targets
        recall, precision, f1 = f1_score(
            tar_paths, pre_paths, tag, self.tag_map)
        # recall, precision, f1 = new_f1_score(
        #     tar_paths, pre_paths, tag, self.tag_map)
        best = self.best_dev_f1.eval()
        if f1 > best:
            print("\tnew best f1:")
            print("\trecall {:.2f}\t precision {:.2f}\t f1 {:.2f}".format(
                recall, precision, f1))
            tf.assign(self.best_dev_f1, f1).eval()
        self.saver.save(sess, self.checkpoint_path)

    def evaluate_test(self, sess, tag):
        result = []
        trans = self.trans.eval()
        batch = self.test_batch.__next__()
        inputs, targets = zip(*batch)
        feed = {
            self.inputs: inputs,
            self.targets: targets,
            self.dropout: 1
        }
        scores, acc, lengths = sess.run(
            [self.logits, self.accuracy, self.length], feed_dict=feed)

        pre_paths = self.decode(scores, lengths, trans)

        tar_paths = targets
        recall, precision, f1 = f1_score(
            tar_paths, pre_paths, tag, self.tag_map)
        # recall, precision, f1 = new_f1_score(
        #     tar_paths, pre_paths, tag, self.tag_map)
        best = self.best_dev_f1.eval()
        if f1 > best:
            print("\tnew best f1:")
            print("\trecall {:.2f}\t precision {:.2f}\t f1 {:.2f}".format(
                recall, precision, f1))
            tf.assign(self.best_dev_f1, f1).eval()

    def prepare_pred_data(self, text):
        vec = [self.vocab.get(i, 0) for i in text]
        feed = {
            self.inputs: [vec],
            self.dropout: 1
        }
        return feed

    def prepare_bert_pred_data(self, text):
        tokens = list(text)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        segment_ids = [0] * len(input_ids)

        feed = {
            self.input_ids: [input_ids],
            self.segment_ids: [segment_ids],
            self.input_mask: [input_mask],
            self.dropout: 1
        }
        return feed

    def predict(self):
        f = open("data/data_map.pkl", "rb")
        maps = cPickle.load(f)
        f.close()
        self.batch_size = 1
        if ARGS.mode == "bert":
            self.tokenizer = tokenization.FullTokenizer(
                vocab_file=ARGS.vocab_dir,
            )
            self.train_length = 10
        else:
            self.vocab = maps.get("vocab", {})
            self.input_size = maps.get("input_size", 10000) + 1

        self.tag_map = maps.get("tag_map", {})
        self.nums_tags = len(self.tag_map.values())
        self.__create_model()
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
            if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
                print("[->] restore model")
                self.saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                print("[->] no model, initializing")
                sess.run(tf.global_variables_initializer())

            trans = self.trans.eval()
            while True:
                text = input(" > ")

                if ARGS.mode == "bert":
                    feed = self.prepare_bert_pred_data(text)
                else:
                    feed = self.prepare_pred_data(text)

                logits, length = sess.run(
                    [self.logits, self.length], feed_dict=feed)
                paths = self.decode(logits, length, trans)
                print(paths)
                org = get_tags(paths[0], "ORG", self.tag_map)
                org_entity = format_result(org, text, "ORG")
                per = get_tags(paths[0], "LOC", self.tag_map)
                per_entity = format_result(per, text, "LOC")

                resp = org_entity["entities"] + per_entity["entities"]
                print(json.dumps(resp, indent=2, ensure_ascii=False))

    def test(self):
        f = open("data/data_map.pkl", "rb")
        maps = cPickle.load(f)
        f.close()
        self.batch_size = 1
        if ARGS.mode == "bert":
            from bert_data_utils import BertDataUtils
            self.tokenizer = tokenization.FullTokenizer(
                vocab_file=ARGS.vocab_dir,
            )
            self.test_data = BertDataUtils(self.tokenizer, batch_size=10)
            self.test_batch = self.test_data.iteration()
            self.train_length = len(self.test_data.data)
        else:
            from data_utils import DataBatch
            self.test_data = DataBatch(data_type='test', batch_size=120)
            self.test_batch = self.test_data.iteration()
            self.vocab = maps.get("vocab", {})
            self.input_size = maps.get("input_size", 10000) + 1

        self.tag_map = maps.get("tag_map", {})
        self.nums_tags = len(self.tag_map.values())
        self.__creat_model()
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
            if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
                print("[->] restore model")
                self.saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                print("[->] no model, initializing")
                sess.run(tf.global_variables_initializer())

            trans = self.trans.eval()
            for batch in self.test_data.get_batch():
                if ARGS.mode == "bert":
                    global_steps, loss, logits, acc, length = self.bert_step(
                        sess, batch
                    )
                else:
                    global_steps, loss, logits, acc, length = self.step(
                        sess, batch
                    )
            if ARGS.mode == "bert":
                self.bert_evaluate(sess, "ORG")
                self.bert_evaluate(sess, "LOC")
            else:
                self.evaluate_test(sess, "ORG")
                self.evaluate_test(sess, "LOC")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("ChineseNer:\n1.train\t\tTraining the model\n2.predict\tTest the model")
        exit()
    model = Model()
    if ARGS.entry == "train":
        model.train()
    elif ARGS.entry == "predict":
        model.predict()
    elif ARGS.entry == 'test':
        model.test()
