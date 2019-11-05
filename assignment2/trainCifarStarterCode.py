from scipy import misc
import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt
import matplotlib as mp

# --------------------------------------------------
# setup
def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
          stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

def weight_variable(shape):
    '''
    Initialize weights
    :param shape: shape of weights, e.g. [w, h ,Cin, Cout] where
    w: width of the filters
    h: height of the filters
    Cin: the number of the channels of the filters
    Cout: the number of filters
    :return: a tensor variable for weights with initial values
    '''

    # IMPLEMENT YOUR WEIGHT_VARIABLE HERE
    init = tf.truncated_normal(shape,stddev = 0.1)
    W = tf.Variable(init)
    return W

def bias_variable(shape):
    '''
    Initialize biases
    :param shape: shape of biases, e.g. [Cout] where
    Cout: the number of filters
    :return: a tensor variable for biases with initial values
    '''

    # IMPLEMENT YOUR BIAS_VARIABLE HERE
    init = tf.constant(0.1,shape = shape)
    b = tf.Variable(init)
    return b

def conv2d(x, W):
    '''
    Perform 2-D convolution
    :param x: input tensor of size [N, W, H, Cin] where
    N: the number of images
    W: width of images
    H: height of images
    Cin: the number of channels of images
    :param W: weight tensor [w, h, Cin, Cout]
    w: width of the filters
    h: height of the filters
    Cin: the number of the channels of the filters = the number of channels of images
    Cout: the number of filters
    :return: a tensor of features extracted by the filters, a.k.a. the results after convolution
    '''

    # IMPLEMENT YOUR CONV2D HERE
    h_conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    return h_conv

def max_pool_2x2(x):
    '''
    Perform non-overlapping 2-D maxpooling on 2x2 regions in the input data
    :param x: input data
    :return: the results of maxpooling (max-marginalized + downsampling)
    '''

    # IMPLEMENT YOUR MAX_POOL_2X2 HERE
    h_max =  tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')
    return h_max


def Construct_ConvLayer(x_image,dimension,firstLayer = False):
    W_conv = weight_variable(dimension)
    b_conv = bias_variable([dimension[-1]])
    h_conv = tf.nn.relu(conv2d(x_image, W_conv) + b_conv)
    return max_pool_2x2(h_conv), W_conv,h_conv


def Construct_FCLayer(h_pool2, dimension,softmax = False):
    W_fc = weight_variable(dimension)
    b_fc = bias_variable([dimension[-1]])
    h_pool_flat = tf.reshape(h_pool2, [-1, dimension[0]])
    h_fc = tf.matmul(h_pool_flat, W_fc) + b_fc
    if not softmax:
        h_fc = tf.nn.relu(h_fc)
    return h_fc


ntrain =  1000# per class
ntest =  100# per class
nclass =  10# number of classes
imsize = 28
nchannels = 1
batchsize = 1000


Train = np.zeros((ntrain*nclass,imsize,imsize,nchannels))
Test = np.zeros((ntest*nclass,imsize,imsize,nchannels))
LTrain = np.zeros((ntrain*nclass,nclass))
LTest = np.zeros((ntest*nclass,nclass))

itrain = -1
itest = -1
for iclass in range(0, nclass):
    for isample in range(0, ntrain):
        path = './CIFAR10/Train/%d/Image%05d.png' % (iclass,isample)
        im = misc.imread(path); # 28 by 28
        im = im.astype(float)/255
        itrain += 1
        Train[itrain,:,:,0] = im
        LTrain[itrain,iclass] = 1 # 1-hot lable
    for isample in range(0, ntest):
        path = './CIFAR10/Test/%d/Image%05d.png' % (iclass,isample)
        im = misc.imread(path); # 28 by 28
        im = im.astype(float)/255
        itest += 1
        Test[itest,:,:,0] = im
        LTest[itest,iclass] = 1 # 1-hot lable

plt.imshow(Train[-1,:,:,0],cmap = 'gray')
plt.show()
sess = tf.InteractiveSession()

tf_data = tf.placeholder(tf.float32, shape=[None, imsize, imsize, nchannels])   #tf variable for the data, remember shape is [None, width, height, numberOfChannels]
tf_labels = tf.placeholder(tf.float32, shape=[None, nclass]) #tf variable for labels

"""Start Implementing LeNet5"""
# --------------------------------------------------
# model
#create your model
dim1 = [5,5,1,32]
dim2 = [5,5,32,64]
dimdl = [7*7*64,1024]
dimsoft = [1024,10]
# first convolutional layer
h_pool1, W_conv1,h_conv1 = Construct_ConvLayer(tf_data,dim1, firstLayer = True)

# second convolutional layer
h_pool2, W_conv2,h_conv2  = Construct_ConvLayer(h_pool1,dim2)

# densely connected layer
h_fc1 = Construct_FCLayer(h_pool2,dimdl)

# dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# softmax layer
y_conv = Construct_FCLayer(h_fc1_drop,dimsoft,softmax = True)
# --------------------------------------------------
# loss
#set up the loss, optimization, evaluation, and accuracy

#learningRate = [1e-4,1e-3,5e-4,1e-2]
learningRate = [1e-3]

Loss_list = {1e-3:[]}
Accuracy_list = {1e-3:[]}
#TAccuracy_list = {1e-3:[]}
#Loss_list = {1e-4:[],1e-3:[],5e-4:[],1e-2:[]}
#Accuracy_list = {1e-4:[],1e-3:[],5e-4:[],1e-2:[]}
#TAccuracy_list = {1e-4:[],1e-3:[],5e-4:[],1e-2:[]}
for lr in learningRate:
    y = tf.nn.softmax(y_conv, name = 'y')
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(tf_labels * tf.log(y), reduction_indices = [1]),name = 'Loss')
    optimizer = tf.train.AdamOptimizer(lr).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(tf_labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32,name = 'Correct_Prediection'), name = 'accuracy')

    # --------------------------------------------------
    # optimization

    sess.run(tf.initialize_all_variables())
    batch_xs = np.zeros((batchsize, imsize, imsize, nchannels)) #setup as [batchsize, width, height, numberOfChannels] and use np.zeros()
    batch_ys = np.zeros((batchsize, nclass))#setup as [batchsize, the how many classes]


    for i in range(1200): # try a small iteration size once it works then continue
        perm = np.arange(nclass * ntrain)
        np.random.shuffle(perm)
        for j in range(batchsize):
            batch_xs[j,:,:,:] = Train[perm[j],:,:,:]
            batch_ys[j,:] = LTrain[perm[j],:]
        train_accuracy = accuracy.eval(feed_dict = {tf_data: batch_xs, tf_labels: batch_ys, keep_prob: 1.0})
        train_loss = cross_entropy.eval(feed_dict = {tf_data: batch_xs, tf_labels: batch_ys, keep_prob: 1.0})
        #Accuracy_list.append(train_accuracy)
        #Loss_list.append(train_loss)
        Accuracy_list[lr].append(train_accuracy)
        Loss_list[lr].append(train_loss)
        W_firstLayer = W_conv1.eval()
        if i%100 == 0:
            print("step: {} with lr: {}, train accuracy:{}, loss:{}".format(i,lr,train_accuracy,train_loss))

        optimizer.run(feed_dict={tf_data: batch_xs, tf_labels: batch_ys, keep_prob: 0.5}) # dropout only during training

        #TAccuracy_list[lr].append(test_accuracy)

    # --------------------------------------------------
    # test
    Activation_firstL= np.array([h_conv1.eval(feed_dict = {tf_data: Test, tf_labels: LTest, keep_prob: 1.0})])
    Activation_secL= np.array(h_conv2.eval(feed_dict = {tf_data: Test, tf_labels: LTest, keep_prob: 1.0}))
    test_accuracy = accuracy.eval(feed_dict = {tf_data: Test, tf_labels: LTest, keep_prob: 1.0})
    print("test accuracy %g" % test_accuracy)

    print('FirstLayer activation stat: Min: {},Max :{},Mean:{},std:{},var:{}'.format(Activation_firstL.min(),
                                                                                     Activation_firstL.max(),
                                                                                     Activation_firstL.mean(),
                                                                                     Activation_firstL.std(),
                                                                                     Activation_firstL.var()))
    print('SecondLayer activation stat: Min: {},Max :{},Mean:{},std:{},var:{}'.format(Activation_secL.min(),
                                                                                     Activation_secL.max(),
                                                                                     Activation_secL.mean(),
                                                                                     Activation_secL.std(),
                                                                                     Activation_secL.var()))
sess.close()
fig,ax = plt.subplots()
fig.suptitle('Accuracy with different LR AdagradOptimizer',fontsize = 18)
#ax.plot(range(len(Accuracy_list[1e-4])),Accuracy_list[1e-4],'k',label = 'Train Accuracy with LR,1e-4 ')
ax.plot(range(len(Accuracy_list[1e-3])),Accuracy_list[1e-3],'k--',label = 'Train Accuracy with LR,1e-3 ')
#ax.plot(range(len(Accuracy_list[5e-4])),Accuracy_list[5e-4],'r--',label = 'Train Accuracy with LR,5e-4 ')
#ax.plot(range(len(Accuracy_list[1e-2])),Accuracy_list[1e-2],'r',label = 'Train Accuracy with LR,1e-2 ')
ax.set(xlabel='Iteration', ylabel='Accuracy')
ax.legend(loc = 'lower right',shadow = True)
plt.show()
fig.savefig('Accuracy_AdagradOptimizer.png')

fig,bx = plt.subplots()
fig.suptitle('Loss with different LR AdagradOptimizer',fontsize = 18)
#bx.plot(range(len(Loss_list[1e-4])),Loss_list[1e-4],'k',label = 'Train Loss with LR,1e-4 ')
bx.plot(range(len(Loss_list[1e-3])),Loss_list[1e-3],'k--',label = 'Train Loss with LR,1e-3 ')
#bx.plot(range(len(Loss_list[5e-4])),Loss_list[5e-4],'r--',label = 'Train Loss with LR,5e-4 ')
#bx.plot(range(len(Loss_list[1e-2])),Loss_list[1e-2],'r',label = 'Train Loss with LR,1e-2 ')
bx.set(xlabel='Iteration', ylabel='Loss')
bx.legend(loc = 'upper right',shadow = True)
plt.show()
fig.savefig('Loss_AdagradOptimizer.png')

"""fig,cx = plt.subplots()
fig.suptitle('Test Accuracy AdagradOptimizer',fontsize = 18)
cx.plot(range(len(TAccuracy_list[1e-4])),TAccuracy_list[1e-4],'k',label = 'Test with LR,1e-4 ')
cx.plot(range(len(TAccuracy_list[1e-3])),TAccuracy_list[1e-3],'k--',label = 'Test with LR,1e-3 ')
cx.plot(range(len(TAccuracy_list[5e-4])),TAccuracy_list[5e-4],'r--',label = 'Test with LR,5e-4 ')
cx.plot(range(len(TAccuracy_list[1e-2])),TAccuracy_list[1e-2],'r',label = 'Test with LR,1e-2 ')
cx.set(xlabel='Iteration', ylabel='Loss')
cx.legend(loc = 'lower right',shadow = True)
plt.show()
fig.savefig('Test_AdagradOptimizer.png')
"""


fig = plt.figure()
fig.suptitle('Visualize First Conv Layer',fontsize = 18)
for i in range(32):
    ax = fig.add_subplot(4, 8, 1 + i)
    ax.imshow(W_firstLayer[:, :, 0, i], cmap='gray')
    plt.axis('off')
plt.show()
#fig.savefig('FistConv.png')