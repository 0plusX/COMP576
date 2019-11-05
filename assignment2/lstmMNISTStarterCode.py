import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.ops import rnn, rnn_cell
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True) #call mnist function

learningRate = 8e-3
trainingIters = 60000
batchSize = 128
displayStep = 10

nInput = 28#we want the input to take the 28 pixels
nSteps = 28#every 28
#nHidden = 128#number of neurons for the RNN
nClasses = 10#this is MNIST so you know

#x = tf.placeholder('float', [None, nSteps, nInput])
#y = tf.placeholder('float', [None, nClasses])




def RNN(x, weights, biases,nHidden,Method):
	x = tf.transpose(x, [1,0,2])
	x = tf.reshape(x, [-1, nInput])
	x = tf.split(x, nSteps, 0) #configuring so you can get it as needed for the 28 pixels

	if Method == 'lstm':
		lstmCell =  tf.contrib.rnn.LSTMCell(nHidden)#find which lstm to use in the documentation
		outputs, states = tf.contrib.rnn.static_rnn(cell = lstmCell, inputs = x,dtype = tf.float32)  # for the rnn where to get the output and hidden state
		return tf.matmul(outputs[-1], weights['out']) + biases['out']
	elif Method == 'gru':
		gruCell = tf.contrib.rnn.GRUCell(nHidden)
		outputs, states = tf.contrib.rnn.static_rnn(cell = gruCell, inputs = x,dtype = tf.float32)  # for the rnn where to get the output and hidden state
		return tf.matmul(outputs[-1], weights['out']) + biases['out']
	elif Method == 'rnn':
		rnnCell = tf.contrib.rnn.BasicRNNCell(nHidden)
		outputs, states = tf.contrib.rnn.static_rnn(cell = rnnCell, inputs = x,dtype = tf.float32)  # for the rnn where to get the output and hidden state
		return tf.matmul(outputs[-1], weights['out']) + biases['out']


def train(method , nHidden):
	x = tf.placeholder('float', [None, nSteps, nInput])
	y = tf.placeholder('float', [None, nClasses])

	biases = {
		'out': tf.Variable(tf.random_normal([nClasses]))
	}

	weights = {
		'out': tf.Variable(tf.random_normal([nHidden, nClasses]))
	}
	pred = RNN(x, weights, biases,nHidden,method)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = pred, labels = y))
	optimizer = tf.train.GradientDescentOptimizer(learningRate).minimize(cost)

	correctPred = tf.equal(tf.argmax(y, 1), tf.argmax(pred, 1))
	accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32, name = 'Correct_Prediection'), name = 'accuracy')

	init = tf.initialize_all_variables()
	# optimization
	# create the cost, optimization, evaluation, and accuracy
	# for the cost softmax_cross_entropy_with_logits seems really good
	Loss_list = []
	Accuracy_list = []
	with tf.Session() as sess:
		sess.run(init)
		step = 1

		while step * batchSize < trainingIters:
			batchX, batchY = mnist.train.next_batch(batchSize)  # mnist has a way to get the next batch
			batchX = batchX.reshape((batchSize, nSteps, nInput))

			sess.run(optimizer, feed_dict = {x: batchX, y: batchY})
			acc = sess.run(accuracy, feed_dict = {x: batchX, y: batchY})
			loss = sess.run(cost, feed_dict = {x: batchX, y: batchY})
			Accuracy_list.append(acc)
			Loss_list.append(loss)
			if step % displayStep == 0:

				print("Iter " + str(step * batchSize) + ", Minibatch Loss= " + \
				      "{:.6f}".format(loss) + ", Training Accuracy= " + \
				      "{:.5f}".format(acc))
			step += 1
		print('Optimization finished')

		testData = mnist.test.images.reshape((-1, nSteps, nInput))
		testLabel = mnist.test.labels
		Test_Accuracy = sess.run(accuracy, feed_dict = {x:testData, y:testLabel})
		print("Testing Accuracy: {}".format(Test_Accuracy) )
	sess.close()
	return Accuracy_list,Loss_list,Test_Accuracy

#rnn_acc, rnn_loss = [], []
#lstm_acc,lstm_loss = [], []
#gru_acc,gru_loss = [], []
#rnn_test,lstm_test,gru_test = 0,0,0

rnn_acc, rnn_loss, rnn_test = train('rnn',128)
lstm_acc, lstm_loss,lstm_test = train('lstm',128)
gru_acc, gru_loss,gru_test = train('gru',128)
fig, ax = plt.subplots()
fig.suptitle('Same Hiddenlayers,Different Method', fontsize = 18)
ax.plot(range(len(rnn_acc)),rnn_acc,'k',label = 'Train Accuracy with RNN ')
ax.plot(range(len(lstm_acc)), lstm_acc, 'g', label = 'Train Accuracy with LSTM ')
ax.plot(range(len(gru_acc)),gru_acc,'r',label = 'Train Accuracy with GRU ')
ax.set(xlabel = 'Iteration', ylabel = 'Accuracy')
ax.legend(loc = 'lower right', shadow = True)
plt.show()
fig.savefig('DifferentMethodACC.png')

fig, bx = plt.subplots()
fig.suptitle('Same Hiddenlayers,Different Method', fontsize = 18)
bx.plot(range(len(rnn_loss)),rnn_loss,'k',label = 'Train LOSS with RNN ')
bx.plot(range(len(lstm_loss)), lstm_loss, 'g', label = 'Train LOSS with LSTM ')
bx.plot(range(len(gru_loss)),gru_loss,'r',label = 'Train LOSS with GRU ')
ax.set(xlabel = 'Iteration', ylabel = 'Loss')
bx.legend(loc = 'upper right', shadow = True)
plt.show()
fig.savefig('DifferentMethodLOSS.png')

print('Test Accuracy: RNN:{}, LSTM:{},GRU :{}'.format(rnn_test,lstm_test,gru_test))



"""acc128, loss128, test128 = train('rnn',128)
fig, ax = plt.subplots()
fig.suptitle('128 Hiddenlayers with basicRNN', fontsize = 18)
ax.plot(range(len(acc128)), acc128, 'k', label = 'Train Accuracy with 128 ')
ax.plot(range(len(loss128)), loss128, 'k', label = 'Train LOSS with 128 ')
ax.set(xlabel = 'Iteration', ylabel = 'Accuracy & Loss')
ax.legend(loc = 'best', shadow = True)
plt.show()
fig.savefig('DifferentLayer128.png')
print('Test Accuracy: 128: {}'.format(test128))"""
