#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import os
import time

'''
This code implements a vae with Gaussian prior and Bernoulli likelihood based on zhusuan
    * the framework has been setup, please only fill the space with "TODO" comments
    * typically, one line of code is sufficient for one "TODO" comment
    * you may see some detailed instructions in the comments, for detailed usage of zhusuan, please see the documentation on http://zhusuan.readthedocs.io/en/latest/concepts.html
'''

'''
Import libs including tensorflow and zhusuan
'''
import tensorflow as tf
from tensorflow.contrib import layers
from six.moves import range
import numpy as np
import zhusuan as zs

import conf
import dataset
import matplotlib
matplotlib.use('tkAgg')
import matplotlib.pyplot as plt
from utils import save_image_collections

'''
Define the generative model according to the generative process
'''
@zs.reuse('model')
def vae(observed, n, n_x, n_z, n_samples, is_training):
    with zs.BayesianNet(observed=observed) as model:
        normalizer_params = {'is_training': is_training,
                             'updates_collections': None}

        z_mean = tf.zeros([n, n_z]) # the mean of z is the zero vector
        z_std = 1. # the covariance of z is the identity matrix, here a scalar is sufficient because it will be broadcasted to a vector and then used as the diagonal of the covariance matrix in zhusuan
        z = zs.Normal('z', mean=z_mean, std=z_std, n_samples=n_samples, group_event_ndims=1)
        '''
        TODO1: sampling z using the Gaussian distribution of zhusuan
            > given input
                - z_mean, z_std, n_samples
                - set group_event_ndims as 1
            > e.g.
                - x = zs.Bernoulli('x', mu, group_event_ndims=1)
        '''


        lx_z_1 = layers.fully_connected(
            z, 500, normalizer_fn=layers.batch_norm,
            normalizer_params=normalizer_params) # a mlp layer of 500 hidden units with z as the input
        lx_z_2 = layers.fully_connected(
            lx_z_1, 500, normalizer_fn=layers.batch_norm,
            normalizer_params=normalizer_params)
        '''
        TODO2: add one more mlp layer with 500 hidden units
            > given input 
                - lx_z_1, size of hidden units, normalizer_params
                - add batch_norm as the normalizer_fn
            > e.g.
                see the above line
        '''

        x_logits = layers.fully_connected(lx_z_2, n_x, activation_fn=None)
        x = zs.Bernoulli('x', x_logits, group_event_ndims=1)
    return model, x_logits


'''
Define the recognition model
'''
@zs.reuse('variational')
def q_net(observed, x, n_z, n_samples, is_training):
    with zs.BayesianNet(observed=observed) as variational:
        normalizer_params = {'is_training': is_training,
                             'updates_collections': None}
        x = tf.to_float(x) # for computation issue
        lz_x_1 = layers.fully_connected(
            x, 500, normalizer_fn=layers.batch_norm,
            normalizer_params=normalizer_params)  # a mlp layer of 500 hidden units with z as the input
        lz_x_2 = layers.fully_connected(
            lz_x_1, 500, normalizer_fn=layers.batch_norm,
            normalizer_params=normalizer_params)
        '''
        TODO3: add two more mlp layers with 500 hidden units
            > given input 
                - x, size of hidden units, normalizer_params
                - add batch_norm as the normalizer_fn
            > e.g.
                see the generative model
        '''
        

        z_mean = layers.fully_connected(lz_x_2, n_z, activation_fn=None) # compute the mean
        z_logstd = layers.fully_connected(lz_x_2, n_z, activation_fn=None) # compute the log std
        z = zs.Normal('z', mean=z_mean, logstd=z_logstd, n_samples=n_samples, group_event_ndims=1)
        '''
        TODO4: sampling z using the Gaussian distribution of zhusuan
            > given input
                - z_mean, z_logstd (note that it is not std), n_samples
                - set group_event_ndims as 1
            > e.g.
                - x = zs.Bernoulli('x', mu, group_event_ndims=1)
        '''

    return variational


if __name__ == "__main__":
    tf.set_random_seed(1237)

    # Load MNIST
    data_path = os.path.join(conf.data_dir, 'mnist.pkl.gz')
    x_train, t_train, x_valid, t_valid, x_test, t_test = \
        dataset.load_mnist_realval(data_path)
    x_train = np.vstack([x_train, x_valid]).astype('float32')
    np.random.seed(1234)
    x_test = np.random.binomial(1, x_test, size=x_test.shape).astype('float32')
    n_x = x_train.shape[1]

    # Define model parameters
    n_z = 40

    # Define training/evaluation parameters
    lb_samples = 10
    ll_samples = 1000
    epochs = 10  # 3000
    batch_size = 100
    iters = x_train.shape[0] // batch_size
    learning_rate = 0.001
    anneal_lr_freq = 200
    anneal_lr_rate = 0.75
    test_freq = 100
    test_batch_size = 400
    test_iters = x_test.shape[0] // test_batch_size
    save_freq = 100
    result_path = "results/vae"

    # Build the computation graph
    is_training = tf.placeholder(tf.bool, shape=[], name='is_training')
    n_particles = tf.placeholder(tf.int32, shape=[], name='n_particles')
    x_orig = tf.placeholder(tf.float32, shape=[None, n_x], name='x')
    x_bin = tf.cast(tf.less(tf.random_uniform(tf.shape(x_orig), 0, 1), x_orig),
                    tf.int32)
    x = tf.placeholder(tf.int32, shape=[None, n_x], name='x')
    x_obs = tf.tile(tf.expand_dims(x, 0), [n_particles, 1, 1])
    # x_obs = tf.to_float(x_obs)
    n = tf.shape(x)[0]

    def log_joint(observed):
        model, _ = vae(observed, n, n_x, n_z, n_particles, is_training)
        log_pz, log_px_z = model.local_log_prob(['z', 'x'])
        return log_pz + log_px_z


    variational = q_net({}, x, n_z, n_particles, is_training)
    qz_samples, log_qz = variational.query('z', outputs=True,
                                           local_log_prob=True)
    lower_bound = tf.reduce_mean(
        zs.sgvb(log_joint, {'x': x_obs}, {'z': [qz_samples, log_qz]}, axis=0))
    '''
        TODO5: compute the lowerbound of VAEs based on zhusuan
            > e.g.    
                - is_log_likelihood = tf.reduce_mean(zs.is_loglikelihood(log_joint, {'x': x_obs},
                            {'z': [qz_samples, log_qz]}, axis=0))
                - the sgvb estimator is built in zhusuan and you can just use it directly instead of using is_loglikelihood
                - the result is stored in the variable lower_bound, i.e., you should write
                    lower_bound = ...
    '''
    

    # Importance sampling estimates of marginal log likelihood
    is_log_likelihood = tf.reduce_mean(
        zs.is_loglikelihood(log_joint, {'x': x_obs},
                            {'z': [qz_samples, log_qz]}, axis=0))

    learning_rate_ph = tf.placeholder(tf.float32, shape=[], name='lr')
    optimizer = tf.train.AdamOptimizer(learning_rate_ph, epsilon=1e-4)
    grads = optimizer.compute_gradients(-lower_bound)
    infer = optimizer.apply_gradients(grads)

    n_gen = 100
    _, x_logits = vae({}, n_gen, n_x, n_z, 1, False)
    x_gen = tf.reshape(tf.sigmoid(x_logits), [-1, 28, 28, 1])

    params = tf.trainable_variables()
    for i in params:
        print(i.name, i.get_shape())

    saver = tf.train.Saver(max_to_keep=10)

    # Run the inference
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)  # use only part of the gpu

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        sess.run(tf.global_variables_initializer())

        # Restore from the latest checkpoint
        ckpt_file = tf.train.latest_checkpoint(result_path)
        begin_epoch = 1
        if ckpt_file is not None:
            print('Restoring model from {}...'.format(ckpt_file))
            begin_epoch = int(ckpt_file.split('.')[-2]) + 1
            saver.restore(sess, ckpt_file)

        for epoch in range(begin_epoch, epochs + 1):
            time_epoch = -time.time()
            if epoch % anneal_lr_freq == 0:
                learning_rate *= anneal_lr_rate
            np.random.shuffle(x_train)
            lbs = []
            for t in range(iters):
                x_batch = x_train[t * batch_size:(t + 1) * batch_size]
                x_batch_bin = sess.run(x_bin, feed_dict={x_orig: x_batch})
                _, lb = sess.run([infer, lower_bound],
                                 feed_dict={x: x_batch_bin,
                                            learning_rate_ph: learning_rate,
                                            n_particles: lb_samples,
                                            is_training: True})
                lbs.append(lb)
            time_epoch += time.time()
            print('Epoch {} ({:.1f}s): Lower bound = {}'.format(
                epoch, time_epoch, np.mean(lbs)))

            images = sess.run(x_gen)
            name = "../results/vae_epoch_{}.png".format(epoch)
            save_image_collections(images, name)

            if epoch == 10:
                print('For this assignment, training 10 epochs is sufficient.')
                exit()

            if epoch % test_freq == 0:
                time_test = -time.time()
                test_lbs = []
                test_lls = []
                for t in range(test_iters):
                    test_x_batch = x_test[t * test_batch_size:
                                          (t + 1) * test_batch_size]
                    test_lb = sess.run(lower_bound,
                                       feed_dict={x: test_x_batch,
                                                  n_particles: lb_samples,
                                                  is_training: False})
                    test_ll = sess.run(is_log_likelihood,
                                       feed_dict={x: test_x_batch,
                                                  n_particles: ll_samples,
                                                  is_training: False})
                    test_lbs.append(test_lb)
                    test_lls.append(test_ll)
                time_test += time.time()
                print('>>> TEST ({:.1f}s)'.format(time_test))
                print('>> Test lower bound = {}'.format(np.mean(test_lbs)))
                print('>> Test log likelihood (IS) = {}'.format(
                    np.mean(test_lls)))

            if epoch % save_freq == 0:
                print('Saving model...')
                save_path = os.path.join(result_path,
                                         "vae.epoch.{}.ckpt".format(epoch))
                if not os.path.exists(os.path.dirname(save_path)):
                    os.makedirs(os.path.dirname(save_path))
                saver.save(sess, save_path)
                print('Done')

        # images = sess.run(x_gen)
        # name = "../results/vae.epoch.{}.png".format(epoch)
        # save_image_collections(images, name)
        # print(images.shape)
        # for i in range(100):
        #     gen_images = images[i].reshape(28, 28)
        #     plt.subplot(10, 10, i+1)
        #     plt.imshow(gen_images)
        # plt.show()
        # test_x = np.random.randn(1, 784)
        # gen_samples = sess.run(qx_samples,
        #                        feed_dict={
        #                            # x: test_x,
        #                            n_particles: 100,
        #                            is_training: False
        #                        })
        # print(gen_samples.shape)
        # for i in range(100):
        #     gen_images = gen_samples[i].reshape(28, 28)
        #     plt.subplot(10, 10, i+1)
        #     plt.imshow(gen_images)
        # plt.show()
