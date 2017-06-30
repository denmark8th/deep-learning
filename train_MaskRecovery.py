import tensorflow as tf
import tensorlayer as tl
import numpy as np
import time
import matplotlib.pyplot as plt
import vgg
import CNNmodel
from mpl_toolkits.axes_grid1 import make_axes_locatable

'''
load training image data
batch_size: batch size for each training epoch, default=50
vali_size: how much validation data you wanna put in training, in range [0,1], default=0.1
'''
def load_data(batch_size=50, vali_size=0.2):
    #Training data
    train_optical_path  = 'TRref_2ch_adc2x_pad_train.npy'#'raster_train_all.npy''TRref_L_1pixel_train_all.npy'#r'/home/rapid/Workspace/Training_place/Winnie_tensorlayer/Winnie0121/ref_train_0.5UR.npy'
    train_raster_path = 'Raster_adc2x_pad_train.npy'#'raster_train_all.npy'#r'/home/rapid/Workspace/Training_place/Winnie_tensorlayer/Winnie0121/raster_train.npy'

    print('Loading data...')
    train_opticalImgSeq  = np.load(train_optical_path)
    print('Num of optical images:',train_opticalImgSeq.shape[0])
    train_rasterImgSeq = np.load(train_raster_path)
    print('Num of raster images:', train_rasterImgSeq.shape[0])
    assert train_opticalImgSeq.shape[0] == train_rasterImgSeq.shape[0]

    num_total = train_opticalImgSeq.shape[0]
    num_train = int(num_total*(1-vali_size))
    num_train = int(num_train/batch_size)*batch_size
    num_vali = int(num_total*vali_size)
    num_vali = int(num_vali/batch_size)*batch_size
    train_optical = train_opticalImgSeq[0:num_train,:,:,:].astype(np.float32)
    train_raster = train_rasterImgSeq[0:num_train,:,:,:].astype(np.float32)
    vali_optical = train_opticalImgSeq[num_total-num_vali:,:,:,:].astype(np.float32)
    vali_raster = train_rasterImgSeq[num_total-num_vali:,:,:,:].astype(np.float32)

    return train_optical, train_raster, vali_optical, vali_raster


'''Define the perceptual loss by using VGG network'''
def compute_content_loss(content_layers, net):
    content_loss = 0
    for layer in content_layers:
        generated_images, content_images = tf.split(0, 2, net[layer])
        size = tf.size(generated_images)
        content_loss += tf.nn.l2_loss(generated_images -
                                      content_images) / tf.to_float(size)
    content_loss = content_loss / len(content_layers)

    return content_loss

'''Define image thresholding'''
def img_threshold(img,thres):
    rows, cols = img.shape
    for i in range(rows):
        for j in range(cols):
            if (img[i, j] < thres):
                img[i, j] = 0
            else:
                img[i, j] = 1
    return img

def save_img(predictY, targetY, epoch):
    lenght = predictY.shape[1]
    width = predictY.shape[1]
    cut_bou = 12
    for i in range(0, predictY.shape[0]):
        fig_name = './Output/InputSize_96_96_image_' + str(i) + str(epoch)
        fig = plt.figure()
        plt.subplot(321)
        plt.title(r"predicting, size = %d" % lenght, fontsize="12")
        ax_1 = plt.gca()
        im_1 = ax_1.imshow(predictY[i, cut_bou:lenght-cut_bou, cut_bou:width-cut_bou])
        plt.axis('off')
        divider = make_axes_locatable(ax_1)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im_1, cax=cax)

        plt.subplot(322)
        plt.title(r"reference, size = %d" % lenght, fontsize="12")
        ax_2 = plt.gca()
        im_2 = ax_2.imshow(targetY[i, cut_bou:lenght-cut_bou, cut_bou:width-cut_bou])
        plt.axis('off')
        divider = make_axes_locatable(ax_2)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im_2, cax=cax)

        # plt.figure(3)
        plt.subplot(323)
        plt.title(r"diff, size = %d" % (lenght-cut_bou*2), fontsize="12")
        ax_3 = plt.gca()
        diff = predictY[i, cut_bou:lenght-cut_bou, cut_bou:width-cut_bou] - targetY[i, cut_bou:lenght-cut_bou, cut_bou:width-cut_bou]
        im_3 = ax_3.imshow(diff[:, :])
        plt.axis('off')
        divider = make_axes_locatable(ax_3)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im_3, cax=cax)

        # plt.figure(5)
        plt.subplot(325)
        plt.title(r"predicting_thres, size = %d" % (lenght-cut_bou*2), fontsize="12")
        predictY[i, :, :]=img_threshold(predictY[i, :, :],128)
        ax_5 = plt.gca()
        im_5 = ax_5.imshow(predictY[i, cut_bou:lenght-cut_bou, cut_bou:width-cut_bou])
        plt.axis('off')
        divider = make_axes_locatable(ax_5)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im_5, cax=cax)

        # plt.figure(6)
        plt.subplot(326)
        plt.title(r"reference_thres, size = %d" % (lenght-cut_bou*2), fontsize="12")
        targetY[i, :, :]=img_threshold(targetY[i, :, :],128)
        ax_6 = plt.gca()
        im_6 = ax_6.imshow(targetY[i, cut_bou:lenght-cut_bou, cut_bou:width-cut_bou])
        plt.axis('off')
        divider = make_axes_locatable(ax_6)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im_6, cax=cax)

        # plt.figure(3)
        plt.subplot(324)
        plt.title(r"diff, size = %d" % (lenght-cut_bou*2), fontsize="12")
        ax_4 = plt.gca()
        diff = predictY[i, cut_bou:lenght-cut_bou, cut_bou:width-cut_bou] - targetY[i, cut_bou:lenght-cut_bou, cut_bou:width-cut_bou]
        im_4 = ax_4.imshow(diff[:, :])
        plt.axis('off')
        divider = make_axes_locatable(ax_4)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im_4, cax=cax)

        fig.savefig(fig_name)
        fig.clear()

def main():
    # wt_pixel = 3
    # wt_perceptual = 1
    model_file_name = '_MaskRecovery_model.ckpt'
    VGG_PATH = 'imagenet-vgg-verydeep-19.mat'
    Perceptual_Layer = ['relu1_2']
    resume = False# load model, resume from previous checkpoint?
    # resume = True
    batch_size = 100
    vali_size = 0.1
    x_train, y_train, x_vali, y_vali = load_data(batch_size, vali_size)
    print('Training data size:', x_train.shape)
    print('Validation data size:', x_vali.shape)
    n_iter = x_train.shape[0]/batch_size
    n_epoch = 2000
    print('There will be ', n_iter, 'iterations and ', n_epoch, 'epochs.')
    learning_rate = 0.0001
    print('Learning rate: %f' % learning_rate)
    print('Batch size: %d' % batch_size)

    sess=tf.InteractiveSession()
    x = tf.placeholder(tf.float32, shape=[batch_size, 120, 120, 2])   #image size is 96x96
    y_ = tf.placeholder(tf.float32, shape=[batch_size, 120, 120, 1])
    net_in = tl.layers.InputLayer(x, name='input_layer')
    net_out = CNNmodel.conv_layers(net_in)
    x_ = net_out.outputs #generated recovered raster image from the forward transform network

    # y1 = vgg.preprocess(y_)
    # x_1 = vgg.preprocess(x_)
    # net = vgg.net(VGG_PATH, tf.concat(0, [x_1, y1]))
    # perceptual_loss = compute_content_loss(Perceptual_Layer, net)
    # pixel_loss = (tf.reduce_sum(tf.square(x_ - y_)))/x_train.shape[0]
    # loss = wt_pixel*pixel_loss + wt_perceptual*perceptual_loss#Define loss function#

    # pixel_loss_l1 = (tf.reduce_sum(tf.abs(x_ - y_)))/x_train.shape[0]
    pixel_loss_l2 = (tf.reduce_sum(tf.square(x_ - y_)))/x_train.shape[0]
    loss = pixel_loss_l2

    train_params = net_out.all_params
    train_op = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999,
                                      epsilon=1e-08, use_locking=False).minimize(loss, var_list=train_params)
    sess.run(tf.initialize_all_variables())
    if resume:
        state = 1000
        load_file_name = str(state) + model_file_name
        print('Loading existing model...')
        saver = tf.train.Saver()
        saver.restore(sess, load_file_name)
        # model_file_name = '_370MaskRecovery_model.ckpt'
    net_out.print_params()
    net_out.print_layers()

    for epoch in range(n_epoch):
        start_time = time.time()
        train_loss, num = 0, 0
        for x_train_a, y_train_a in tl.iterate.minibatches(
                x_train, y_train, batch_size, shuffle=True):
            feed_dict = {x: x_train_a, y_: y_train_a}
            sess.run(train_op, feed_dict=feed_dict)
            loss_ = sess.run(loss, feed_dict=feed_dict)
            assert not np.isnan(loss_), 'Model diverged with cost = NaN!'
            train_loss += loss_
            num += 1
            # print('Training loss: %f' % (train_loss/num))
        print('Epoch %d of %d took %fs' % (epoch+1, n_epoch, time.time()-start_time))
        print('Training loss: %f' % (train_loss / num))

        vali_loss, num1 = 0, 0
        for x_vali_a, y_vali_a in tl.iterate.minibatches(
                x_vali, y_vali, batch_size, shuffle=True):
            feed_dict = {x: x_vali_a, y_: y_vali_a}
            y_predict, y_target, err = sess.run([x_, y_, loss], feed_dict=feed_dict)
            y_predict = y_predict * 255
            y_target = y_target * 255
            vali_loss += err
            num1 += 1
        print('Validation loss: %f' % (vali_loss / num1))
        if (epoch) % 10 == 0:
            print("Save model " + "!" * 10)
            file_name = str(epoch+1) + model_file_name
            saver = tf.train.Saver()
            saver.save(sess, file_name)

        if (epoch) % 10 == 0:
            save_img(y_predict.reshape([y_predict.shape[0], 120, 120]), y_target.reshape([y_target.shape[0], 120, 120]),
                     epoch+1)


if __name__ == '__main__':
    sess = tf.InteractiveSession()
    sess = tl.ops.set_gpu_fraction(sess, gpu_fraction = .5)
    try:
        """Without image distorting"""
        # main_test_cnn_naive()
        """With image distorting"""
        main()
        tl.ops.exit_tf(sess)   # close sess, tensorboard and nvidia-process
    except KeyboardInterrupt:
        print('\nKeyboardInterrupt')
tl.ops.exit_tf(sess)