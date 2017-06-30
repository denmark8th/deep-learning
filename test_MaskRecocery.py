import tensorflow as tf
import tensorlayer as tl
import CNNmodel
import os
import sys
import numpy as np
import time
from scipy.misc import imread, imresize
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PIL import Image

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

def save_img(predictY, targetY, x_vali, epoch):
    lenght = predictY.shape[1]
    width = predictY.shape[1]
    cut_bou =12
    for i in range(0, predictY.shape[0]):
        fig_name = './Output/InputSize_96_96_image_' + str(i+1) + str(epoch)

        img_name1 = './Output/Recovered_raster_' + str(i+1) + '.png'
        img1 = predictY[i, cut_bou:lenght - cut_bou, cut_bou:width - cut_bou]
        img1 = Image.fromarray(np.uint8(img1))
        img1.save(img_name1)

        img_name2 = './Output/Original_raster_' + str(i+1) + '.png'
        img2 = targetY[i, cut_bou:lenght - cut_bou, cut_bou:width - cut_bou]
        img2 = Image.fromarray(np.uint8(img2))
        img2.save(img_name2)

        fig = plt.figure()
        plt.subplot(321)
        plt.title(r"predicting, size = %d" % (lenght-cut_bou*2), fontsize="12")
        ax_1 = plt.gca()
        im_1 = ax_1.imshow(predictY[i, cut_bou:lenght-cut_bou, cut_bou:width-cut_bou])
        plt.axis('off')
        divider = make_axes_locatable(ax_1)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im_1, cax=cax)

        plt.subplot(322)
        plt.title(r"reference, size = %d" % (lenght-cut_bou*2), fontsize="12")
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



def main_test():
    model_file_name = "1500_MaskRecovery_model.ckpt"
    epoch = 1500
    batch_size = 155
    test_optical_filepath  = 'TRref_adc2x_pad_test_doi.npy'
    test_raster_filepath = 'Raster_adc2x_pad_test_doi.npy'
    print('Loading test data...')
    test_optical = np.load(test_optical_filepath)
    print('Num of optical images:', test_optical.shape[0])
    test_raster = np.load(test_raster_filepath)
    print('Num of raster images:', test_raster.shape[0])
    print('Finish loading test data')

    '''Load pre-trained model and restore the session'''
    sess=tf.InteractiveSession()
    x = tf.placeholder(tf.float32, shape=[batch_size, 120, 120, 2])   #image size is 96x96
    y_ = tf.placeholder(tf.float32, shape=[batch_size, 120, 120, 1])
    net_in = tl.layers.InputLayer(x, name='input_layer')
    net_out = CNNmodel.conv_layers(net_in)
    x_ = net_out.outputs #generated recovered raster image from the forward transform network
    cost = (tf.reduce_sum(tf.square(x_ - y_))) / test_optical.shape[0]

    print("Load existing model " + "!" * 10)
    saver = tf.train.Saver()
    saver.restore(sess, model_file_name)

    vali_loss, n_batch = 0, 0
    for x_vali_a, y_vali_a in tl.iterate.minibatches(
        test_optical, test_raster, batch_size, shuffle=True):
        feed_dict = {x: x_vali_a, y_: y_vali_a}
        y_predict, y_target, err = sess.run([x_, y_, cost], feed_dict=feed_dict)
        y_predict = y_predict*255
        y_target = y_target*255
        vali_loss += err
        n_batch += 1
        save_img(y_predict.reshape([y_predict.shape[0], 120, 120]), y_target.reshape([y_target.shape[0], 120, 120]), epoch)
    print('Testing loss: %f' % (vali_loss / n_batch))


if __name__ == '__main__':
    sess = tf.InteractiveSession()
    sess = tl.ops.set_gpu_fraction(sess, gpu_fraction = .5)
    try:
        """Without image distorting"""
        # main_test_cnn_naive()
        """With image distorting"""
        main_test()

        tl.ops.exit_tf(sess)   # close sess, tensorboard and nvidia-process
    except KeyboardInterrupt:
        print('\nKeyboardInterrupt')
tl.ops.exit_tf(sess)



