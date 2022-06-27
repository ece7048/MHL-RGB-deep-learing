
#Aknowledgement as source initial Author: https://github.com/kamalkraj/Vision-Transformer
#extantion Author: Michail Mamalakis

from __future__ import division, print_function

import logging
import math

import numpy as np
import tensorflow as tf
from ph_pipeline import vit
from fastprogress import master_bar, progress_bar
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
logger = logging.getLogger(__name__)


class TrainerConfig:
    # optimization parameters
    max_epochs = 40
    batch_size = 16
    learning_rate = 1e-3
    # checkpoint settings
    ckpt_path = None

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class Trainer:


    def fix_gpu(self):
        config = ConfigProto()
        config.gpu_options.allow_growth = True
        session = InteractiveSession(config=config)


    def __init__(self, model, model_config, train_dataset, train_dataset_len, test_dataset, test_dataset_len, config):
        self.config=config
        self.train_dataset = train_dataset.batch(config.batch_size)
        options = tf.data.Options()
        options.experimental_optimization.noop_elimination = True
        options.experimental_optimization.apply_default_optimizations = True
        dataset = self.train_dataset.with_options(options)
        self.train_dataset_len = train_dataset_len
        self.test_dataset = test_dataset
        self.test_dataset_len = None
        self.test_dist_dataset = None
        self.fix_gpu()
        self.model1=model
        self.model_config=model_config
        if self.test_dataset:
            self.test_dataset = test_dataset.batch(config.batch_size)
            self.test_dataset_len = test_dataset_len
            self.test_dataset = self.test_dataset.with_options(options)
        self.config = config
        self.tokens = 0
        self.strategy = tf.distribute.OneDeviceStrategy("GPU:0")
        print("GPU number: ",len(tf.config.list_physical_devices('GPU')))
        if len(tf.config.list_physical_devices('GPU')) > 1:
            self.strategy = tf.distribute.MirroredStrategy()

        #with self.strategy.scope():
            #self.model = model(**model_config)
            #self.optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)
            #self.optimizer = tf.keras.optimizers.SGD(learning_rate=config.learning_rate)
            #self.cce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,reduction=tf.keras.losses.Reduction.NONE)
            #self.cce=tf.keras.losses.CategoricalCrossentropy(from_logits=True,reduction=tf.keras.losses.Reduction.NONE)
            #self.cce=tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
            #self.train_dist_dataset = self.strategy.experimental_distribute_dataset(self.train_dataset)
            #if self.test_dataset:
                #self.test_dist_dataset = self.strategy.experimental_distribute_dataset(self.test_dataset)

    def save_checkpoints(self):
        if self.config.ckpt_path is not None:
            self.model.save_weights(self.config.ckpt_path)
    def load_checkpoints(self):
        if self.config.ckpt_path is not None:
            self.model.load_weights(self.config.ckpt_path)

    def train(self):

        train_loss_metric = tf.keras.metrics.Mean('training_loss', dtype=tf.float32)
        test_loss_metric = tf.keras.metrics.Mean('testing_loss', dtype=tf.float32)

        train_accuracy = tf.keras.metrics.Accuracy('training_accuracy', dtype=tf.float32)
        test_accuracy = tf.keras.metrics.Accuracy('testing_accuracy', dtype=tf.float32)
        train_acc1= tf.keras.metrics.AUC(name='train_ROC',curve='ROC')
        train_acc2= tf.keras.metrics.AUC(name='train_PR', dtype=tf.float32,curve='PR')
        train_acc3= tf.keras.metrics.Recall(name='train_recall', dtype=tf.float32)
        train_acc4=tf.keras.metrics.Precision(name='train_prec', dtype=tf.float32)

        test_acc1= tf.keras.metrics.AUC(name='test_ROC', dtype=tf.float32,curve='ROC')
        test_acc2= tf.keras.metrics.AUC(name='test_PR', dtype=tf.float32,curve='PR')
        test_acc3= tf.keras.metrics.Recall(name='trest_recall', dtype=tf.float32)
        test_acc4=tf.keras.metrics.Precision(name='test_prec', dtype=tf.float32)

        @tf.function
        def train_step(dist_inputs):

            def step_fn(inputs):
                X, Y = inputs
                Y2=tf.cast(Y,dtype=tf.float32)
                self.classes=Y.shape[1]
                #print(Y2[1])
                with tf.GradientTape() as tape:
                # training=True is only needed if there are layers with different
                # behavior during training versus inference (e.g. Dropout)
                    self.load_checkpoints()
                    logits = self.model(X,training=True)		
                    num_labels = self.classes
                    label_mask = tf.math.logical_not(Y < 0)
                    label_mask = tf.reshape(label_mask,(-1,))
                    label_mask_b = tf.reshape(label_mask,(-1,num_labels))
                    logits_b=tf.reshape(logits,(-1,num_labels))
                    logits = tf.reshape(logits,(-1,))
                    logits_masked = tf.boolean_mask(logits,label_mask)
                    label_ids = tf.reshape(Y,(-1,))
                    label_ids_masked = tf.boolean_mask(label_ids,label_mask) #label_ids
                    cross_entropy = self.cce(tf.squeeze(label_mask_b), tf.squeeze(logits_b))
                    loss = tf.reduce_sum(cross_entropy) * (1.0 / self.config.batch_size)
                    y_pred = tf.argmax(tf.nn.softmax((logits_b),axis=-1),axis=-1)
                    y_gr=tf.argmax(tf.nn.softmax(Y2,axis=-1),axis=-1)
                    y_gr1=(label_mask_b)
                    y_pred1=(logits_b)
                    train_accuracy.update_state(y_gr,y_pred)
                    train_acc1.update_state(y_gr,y_pred)#label_ids
                    train_acc2.update_state(y_gr,y_pred)#label_ids
                    train_acc3.update_state(y_gr,y_pred)#label_ids
                    train_acc4.update_state(y_gr,y_pred)#label_ids


                grads = tape.gradient(loss, self.model.trainable_variables)
                self.optimizer.apply_gradients(list(zip(grads, self.model.trainable_variables)))
                return cross_entropy
            per_example_losses = self.strategy.run(step_fn, args=(dist_inputs,))        
            sum_loss = self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_example_losses, axis=0)
            mean_loss = sum_loss / self.config.batch_size
            return mean_loss

        @tf.function
        def test_step(dist_inputs):

            def step_fn(inputs):
                X, Y = inputs
                Y2=tf.cast(Y,dtype=tf.float32)
                # training=True is only needed if there are layers with different
                # behavior during training versus inference (e.g. Dropout)
                self.load_checkpoints()
                logits = self.model(X,training=False)
                num_labels = self.classes
                label_mask = tf.math.logical_not(Y < 0)
                label_mask = tf.reshape(label_mask,(-1,))
                label_mask_b = tf.reshape(label_mask,(-1,num_labels))
                logits_b=tf.reshape(logits,(-1,num_labels))
                logits = tf.reshape(logits,(-1,))
                logits_masked = tf.boolean_mask(logits,label_mask)
                label_ids = tf.reshape(Y,(-1,))
                label_ids_masked = tf.boolean_mask(label_ids,label_mask)
                cross_entropy = self.cce(tf.squeeze(label_mask_b), tf.squeeze(logits_b))
                loss = tf.reduce_sum(cross_entropy) * (1.0 / self.config.batch_size)
                y_pred = tf.argmax(tf.nn.softmax(logits_b,axis=-1),axis=-1)
                y_gr=tf.argmax(tf.nn.softmax(Y2,axis=-1),axis=-1)
                y_gr1=(label_mask_b)
                y_pred1=(logits_b)
                test_accuracy.update_state(y_gr,y_pred)#label_ids
                test_acc1.update_state(y_gr,y_pred)#label_ids
                test_acc2.update_state(y_gr,y_pred)#label_ids
                test_acc3.update_state(y_gr,y_pred)#label_ids
                test_acc4.update_state(y_gr,y_pred)#label_ids


                return cross_entropy

            per_example_losses = self.strategy.run(step_fn, args=(dist_inputs,))
            sum_loss = self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_example_losses, axis=0)
            mean_loss = sum_loss / self.config.batch_size
            return mean_loss

        train_pb_max_len = math.ceil(float(self.train_dataset_len)/float(self.config.batch_size))
        test_pb_max_len = math.ceil(float(self.test_dataset_len)/float(self.config.batch_size)) if self.test_dataset else None

        epoch_bar = master_bar(range(self.config.max_epochs))
        with self.strategy.scope():
            self.model = self.model1(**self.model_config)
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate)
            #self.optimizer = tf.keras.optimizers.SGD(learning_rate=self.config.learning_rate,momentum=0.1, nesterov=True)
            self.cce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,reduction=tf.keras.losses.Reduction.NONE)
            #self.cce=tf.keras.losses.CategoricalCrossentropy(from_logits=True,reduction=tf.keras.losses.Reduction.NONE)
            #self.cce=tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
            self.train_dist_dataset = self.strategy.experimental_distribute_dataset(self.train_dataset)
            if self.test_dataset:
                self.test_dist_dataset = self.strategy.experimental_distribute_dataset(self.test_dataset)

            best=100000000
            for epoch in epoch_bar:
                for inputs in progress_bar(self.train_dist_dataset,total=train_pb_max_len,parent=epoch_bar):
                    #print("first input size: ",inputs.shape)
                    loss = train_step(inputs)
                    #print("Loss is : ",loss)
                    self.tokens += tf.reduce_sum(tf.cast(inputs[1]>=0,tf.int32)).numpy()
                    train_loss_metric(loss)
                    epoch_bar.child.comment = f'training loss : {train_loss_metric.result()}'
                print(f"epoch {epoch+1}: train loss {train_loss_metric.result():.5f}. train accuracy {train_accuracy.result():.5f}")
                print(f"epoch {epoch+1}: train_ROC{train_acc1.result():.5f}. train_PR  {train_acc2.result():.5f}. train_recall{train_acc3.result():.5f}. train_prec  {train_acc4.result():.5f}.")
                train_loss_metric.reset_states()
                train_accuracy.reset_states()

                if self.test_dist_dataset:
                    for inputs in progress_bar(self.test_dist_dataset,total=test_pb_max_len,parent=epoch_bar):
                        loss = test_step(inputs)
                        test_loss_metric(loss)
                        epoch_bar.child.comment = f'testing loss : {test_loss_metric.result()}'
                    print(f"epoch {epoch+1}: test loss {test_loss_metric.result():.5f}. test accuracy {test_accuracy.result():.5f}")
                    print(f"epoch {epoch+1}: test_ROC{test_acc1.result():.5f}. test_PR  {test_acc2.result():.5f}. test_recall{test_acc3.result():.5f}. test_prec  {test_acc4.result():.5f}.")
                    test_loss_metric.reset_states()
                    test_accuracy.reset_states()
                if (train_loss_metric.result()<=best):
                    best=train_loss_metric.result()
                    self.save_checkpoints()
