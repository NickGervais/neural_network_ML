import numpy as np
from layer import Layer


class NeuralNet:
    # initializes layers to a list, layers will contain h layers and output layer
    def __init__(self, num_input_nodes, num_hidden_layers, nodes_per_h, num_output_nodes):
        self.init_weights(num_input_nodes, num_hidden_layers,
                          nodes_per_h, num_output_nodes)

    # creates a set of weights for each layer
    def init_weights(self, num_inputs, num_h_layers, nodes_per_h, output_nodes):
        weights = []
        # iterate through number of hidden layers
        for i in range(num_h_layers):
            # create a set of weights with size (#of inputs)x(#of nodes)
            weight_set = [[(2 * np.random.uniform() - 1)
                           for i in range(num_inputs)] for j in range(nodes_per_h)]
            # append that set of weights the overall weights array
            weights.append(weight_set)
            num_inputs = nodes_per_h
        #appends the output layer
        weight_output_set = [[(2*np.random.uniform()-1) for i in range(nodes_per_h)] for j in range(output_nodes)]
        weights.append(weight_output_set)

        self.weights = weights

    # function that passes output through the sigmoid
    def sigmoid(self, output):
        return 1 / (1 + np.exp(-output))

    # iterates through each layer, "inputs" is outputs from previous layer
    def forward_propagate(self, inputs):
        outputs = []
        #convet inputs to a numpy array for dot product functionality
        inputs = np.array([inputs])
        # iterate throught each layer
        for i in range(len(self.weights)):
            # set the outputs of this layer to the dot product of the inputs and the weights
            output_set = inputs.dot(np.transpose(self.weights[i]))
            # append the sigmoid of the output set to the total outputs array
            sig_output_set = self.sigmoid(output_set)
            outputs.append(sig_output_set)
            # the next inputs is the outputs of this layer
            inputs = sig_output_set
        # finally set this neural nets outputs to outputs
        self.outputs = outputs
        return inputs

    def sig_derv(self, output):
        return output * (1 - output)

    # reverse iterate through the layers and calculate delta for each node
    def back_propagate(self, expected_outputs):
        delta = []
        # iterate through layers in reversed order
        for i in reversed(range(len(self.weights))):
            # if this is the output layer then:
            if i == len(self.weights) - 1:
                # output layers delta is expected - calculated
                delta.append(expected_outputs - self.outputs[i])
            # otherwise it is not the output layer
            else:
                #calculate the error by dot product of delta and weights from the proceeding layer
                j = len(self.weights) - i -1
                error_set = delta[j-1].dot(self.weights[i+1])
                #calculate the delta for each layer by error_set times the sigmoid derivative of the outputs
                delta_set = error_set * self.sig_derv(self.outputs[i])
                # append this layers delta the overall delta array
                delta.append(delta_set)
        # reverse the order of the delta b/c it was set in reversed order
        self.delta = delta[::-1]

    # iterate through each layer and update the weights based off the calculated deltas
    def update_weights(self, inputs, learning_rate):
        inputs = np.array([inputs])
        # iterate through each layer
        for i in range(len(self.weights)):
            # if it is not the first hidden layer then:
            if i != 0:
                # update the inputs to the outputs from the previous layer
                inputs = self.outputs[i - 1]
            # updates the weights to the weights plus the learning rate times the dot product of the deltas and inputs of that layer
            self.weights[i] += learning_rate * (np.transpose(self.delta[i]).dot(inputs))

    def softmax(self, x):
        return (np.exp(x)) / (sum(np.exp(x)))

    def train_net(self, training_dataset, test_dataset, learning_rate, num_epochs, num_outputs):
	val_acc = []
	acc = []
	train_size = len(training_dataset)
	test_size = len(test_dataset)
        for epoch in range(num_epochs):
            sum_error = 0
            for data in training_dataset:
                # data[:-1] removes the expected value from the input array
                outputs = self.forward_propagate(data[:-15])[0]
                # the expected output is an array of all 0 except for a 1 in the location specified by the last value in the data array
                expected_outputs = data[-15:]
                sum_error += sum((expected_outputs - outputs)**2)
                #sum_error += sum([(expected_outputs[i]-outputs[i])**2 for i in range(len(expected_outputs))])/len(training_dataset)
		self.back_propagate(expected_outputs)
                self.update_weights(data[:-15], learning_rate)
	    train_correct = 0
	    for img in training_dataset:
	        pred = np.argmax(predict(img[:-15]))
	        actual = np.argmax(img[-15:])
	        if pred == actual:
		    train_correct += 1
	    acc.append(train_correct/train_size)

	    test_correct = 0
	    for img in test_dataset:
		pred = np.argmax(predict(img[:-15]))
		actual = np.argmax(img[-15:])
		if pred == actual:
		    test_correct += 1
	    val_acc.append(test_correct/test_size)
            print('epoch=%d, learning rate=%.3f, error = %.3f' %
                  (epoch, learning_rate, sum_error))
	    print('acc = ', acc[epoch])
	    print('val_acc = ', val_acc[epoch])

	valdata = np.array(val_acc)
	accdata = np.array(acc)
	np.save('val_acc', valdata)
	np.save('acc', accdata)
    # will be called when testing
    def predict(self, image_data):
        # run testing
	outputs = self.forward_propagate(image_data)
	return outputs
