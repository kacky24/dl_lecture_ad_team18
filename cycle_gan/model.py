from __future__ import division

import module
import numpy as np
import time
import os
import keras.backend as K

from glob import glob
from keras.models import Model
from keras.layers import Input, Lambda
from keras.optimizers import Adam

from module import *
from utils import *

class cycleGAN:
    
    def __init__(self, sess, args):
        self.batch_size = args.batch_size
        self.checkpoint_dir = args.checkpoint_dir
        self.data_dir = args.data_dir
        self.epochs　= args.epochs　
        self.L1_lambda = args.L1_lambda # reconstruction errorの強さを調整するパラメータ
        self.max_train_size = args.max_train_size
        self.sample_freq = args.sample_freq
        self.save_freq = args.save_freq
        self.sample_dir = args.sample_dir
        self.test_dir　= args.test_dir 
        self.weight_dir = args.weight_path
        self.which_direction = args.which_direction
        
        self.pool = ImagePool(args.max_pool_size)

        self.generator_A2B = module.generator((None, 256, 256, 3))
        self.generator_B2A = module.generator((None, 256, 256, 3))
        
        self.discriminatorA = module.discriminator((None, 256, 256, 3))
        self.discriminatorB = module.discriminator((None, 256, 256, 3))
        
    def train(self):
        
        counter = 1
        start_time = time.time()
        
        # if there exists, load weights
        if self.weight_path != None:
            self.generator_A2B.load_weights(os.path.join(self.weight_dir, 'generator_A2B.h5'))
            self.generator_B2A.load_weights(os.path.join(self.weight_dir, 'generator_B2A.h5'))
            self.discriminatorA.load_weights(os.path.join(self.weight_dir, 'discriminatorA.h5'))
            self.discriminatorB.load_weights(os.path.join(self.weight_dir, 'discriminatorB.h5'))
            
        self.discriminatorA.compile(loss='mean_squared_error', optimizer=Adam(lr=0.0002, beta1=0.5))
        self.discriminatorB.compile(loss='mean_squared_error', optimizer=Adam(lr=0.0002, beta1=0.5))
        
        self.discriminatorA.trainable = False
        self.discriminatorB.trainable = False
        
        # 入力層
        inputA = Input(shape = (None, 256, 256, 3), name='input_A')
        inputB = Input(shape = (None, 256, 256, 3), name='input_B')
        
        # generatorでドメイン変換
        fakeB = self.generator_A2B(inputA)
        fakeA = self.generator_B2A(inputB)
        # discriminatorのlossを計算するためにnameが欲しい
        fakeB = Lambda(lambda x: x, name='fake_B')
        fakeA = Lambda(lambda x: x, name='fake_A')
        
        # cycle lossを計算するために再びドメイン変換
        reconstA = self.generator_B2A(fakeB)
        reconstB = self.generator_A2B(fakeA)
        
    　　　　# GAN lossを計算するためにdiscriminatorにかける
        discriminatedA = self.discriminatorA(fakeA)
        discriminatedB = self.discriminatorB(fakeB)
        
        # generatorの最適化するためのモデル
        cycleGAN_generator_train = Model(inputs=[inpA, inpB],
                        outputs=[discriminatedA, discriminatedB, reconstA, reconstB])
        
        cycleGAN_generator_train.compile(optimizer=Adam(lr=0.0002, beta1=0.5),
                        loss=['mean_squared_error', 'mean_squared_error', 'mean_absolute_error', 'mean_absolute_error'],
                        loss_weights=[1., 1., self.L1_lambda, self.L1_lambda])
        
        # discriminatorに流すためのfake sample生成器 
        # もしかしてoutputs=[fakeA, fakeB]でよくない？
        generate_fake_samples = Model(inputs=[cycleGAN_generator_train.input,
                                             outputs=[cycleGAN_generator_train.get_layer('fake_A').output, cycleGAN_generator_train.get_layer('fake_B').output]])
                
        for epoch in self.epochs:
            if epoch < 100:
                pass
            else:
                learning_rate = 0.0002*(self.epochs - epoch/self.epochs - 100)
                K.set_value(cycleGAN_generator_train.optimizer.lr, learning_rate)
                K.set_value(self.discriminatorA.optimizer.lr, learning_rate)
                K.set_value(self.discriminatorB.optimizer.lr, learning_rate)
            
            # load data paths and shuffle
            domainA = glob('./' + self.data_dir + '/trainA/*.*')
            domainB = glob('./' + self.data_dir + '/trainB/*.*')
            np.random.shuffle(domainA)
            np.random.shuffle(domainB)
            
            iterations = min(min(len(domainA), len(domainB)), self.max_train_size) // self.batch_size
            
            for i in iterations:
                batchA = []
                batchB = []
                for batch_file in batchfiles:
                    img_A, img_B = load_train_data(batch_file)
                    batchA.append(img_A)
                    batchB.append(img_B)
                batchA = np.array(batchA).astype(np.float32)
                batchB = np.array(batchB).astype(np.float32)
                
                # generatorのトレーニング
                cycleGAN_generator_train.train_on_batch(x=[batchA, batchB],
                                                        y=[K.ones((self.batch_size, 16, 16, 1)), K.ones((self.batch_size, 16, 16, 1)), batchA, batchB])
                
                # discriminatorのトレーニングのためにfakeサンプルを生成
                fakeA_sample, fakeB_sample = generate_fake_samples.predict(x=[batchA, batchB])
                # 確率的にpoolした生成画像と入れ替え
                [fakeA_sample, fakeB_sample] = self.ImagePool([fakeA_sample, fakeB_sample])
                
                self.discriminatorA．train_on_batch(x=batchA, y=K.ones((self.batch_size, 16, 16, 1)))
                self.discriminatorA．train_on_batch(x=fakeA_sample, y=K.zeros((self.batch_size, 16, 16, 1)))
                self.discriminatorB．train_on_batch(x=batchB, y=K.ones((self.batch_size, 16, 16, 1)))
                self.discriminatorB．train_on_batch(x=fakeB_sample, y=K.zeros((self.batch_size, 16, 16, 1)))
                
                counter += 1
                print(("Epoch: [%2d] [%4d/%4d] time: %4.4f" % (
                    epoch, i, iterations, time.time() - start_time)))
                
                if np.mod(counter, self.sample_freq) == 1:
                    domainA = glob('./' + self.data_dir + '/testA/*.*')
                    domainB = glob('./' + self.data_dir + '/testB/*.*')
                    np.random.shuffle(domainA)
                    np.random.shuffle(domainB)
                    
                    batchA = []
                    batchB = []
                    for batch_file in batchfiles:
                        img_A, img_B = load_train_data(batch_file, is_testing=True)
                        batchA.append(img_A)
                        batchB.append(img_B)
                    batchA = np.array(batchA).astype(np.float32)
                    batchB = np.array(batchB).astype(np.float32)
                    
                    fakeB = self.generator_A2B.predict(batchA)
                    fakeA = self.generator_B2A.predict(batchB)
                    
                    save_images(fakeA, [self.batch_size, 1],　'./' + self.sample_dir + 'A_{:02d}_{:04d}.jpg'.format(epoch, i))
                    save_images(fakeB, [self.batch_size, 1],　'./' + self.sample_dir + 'B_{:02d}_{:04d}.jpg'.format(epoch, i))
                
                # ネットワークのパラメータを保存
                if np.mod(counter, self.save_freq) == 2:
                    model_name = "cyclegan.model"
                    model_dir = "%s_%s" % (self.data_dir, 256)
                    checkpoint_dir = os.path.join(self.checkpoint_dir, model_dir)

                    if not os.path.exists(self.checkpoint_dir):
                        os.makedirs(self.checkpoint_dir)
                        
                    self.generator_A2B.save_weights('generator_A2B.h5')
                    self.generator_B2A.save_weights('generator_B2A.h5')
                    self.discriminatorA.save_weights('discriminatorA.h5')
                    self.discriminatorB.save_weights('discriminatorB.h5')
        
    def test(self):
        # 保存した重みを呼び出し
        if self.weight_path != None:
            self.generator_A2B.load_weights(os.path.join(self.weight_dir, 'generator_A2B.h5'))
            self.generator_B2A.load_weights(os.path.join(self.weight_dir, 'generator_B2A.h5'))
        
        # 変換方向を指定
        if self.which_direction == 'AtoB':
            sample_files = glob('./' + self.data_dir + '/testA/*.*')
            if self.weight_path != None:
                self.generator_A2B.load_weights(os.path.join(self.weight_dir, 'generator_A2B.h5'))
                print('Load weights successfully!!')
            else:
                print("Fail to load weights...")
            generator = self.generator_A2B()
        elif self.which_direction == 'BtoA':
            sample_files = glob('./' + self.data_dir + '/testB/*.*')
            if self.weight_path != None:
                self.generator_B2A.load_weights(os.path.join(self.weight_dir, 'generator_B2A.h5'))
                print('Load weights successfully!!')
            else:
                print("Fail to load weights...")
            generator = self.generator_B2A()
        else:
            raise Exception('--which_direction must be AtoB or BtoA')
            
        # 生成した画像をhtmlで比較できるようにする
        index_path = os.path.join(self.test_dir, '{0}_index.html'.format(self.which_direction))
        index = open(index_path, "w")
        index.write("<html><body><table><tr>")
        index.write("<th>name</th><th>input</th><th>output</th></tr>")

        for sample_file in sample_files:
            print('Processing image: ' + sample_file)
            sample = [load_test_data(sample_file, 256)]
            sample = np.array(sample).astype(np.float32)
            image_path = os.path.join(self.test_dir + '{0}_{1}'.format(self.which_direction, os.path.basename(sample_file)))
            
            fake = generator.predict(sample)
            save_images(fake, [1, 1], image_path)
            index.write("<td>%s</td>" % os.path.basename(image_path))
            index.write("<td><img src='%s'></td>" % (sample_file if os.path.isabs(sample_file) else (
                '..' + os.path.sep + sample_file)))
            index.write("<td><img src='%s'></td>" % (image_path if os.path.isabs(image_path) else (
                '..' + os.path.sep + image_path)))
            index.write("</tr>")
        index.close()