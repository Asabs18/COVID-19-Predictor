# import pandas
# import tensorflow
# data = pandas.read_csv('../scraper/data.csv')

# x_train, y_train = data[SOME AMOUNT OF DATA], data[SAOD]
# x_test, y_test = data[SOME AMOUNT OF DATA], data[SAOD]

# # Layer 1: Variables for hidden weights and biases
# W_hidden_1 = tf.Variable(weight_initializer([n_stocks, n_neurons_1]))
# bias_hidden_1 = tf.Variable(bias_initializer([n_neurons_1]))
# # Layer 2: Variables for hidden weights and biases
# W_hidden_2 = tf.Variable(weight_initializer([n_neurons_1, n_neurons_2]))
# bias_hidden_2 = tf.Variable(bias_initializer([n_neurons_2]))
# # Layer 3: Variables for hidden weights and biases
# W_hidden_3 = tf.Variable(weight_initializer([n_neurons_2, n_neurons_3]))
# bias_hidden_3 = tf.Variable(bias_initializer([n_neurons_3]))
# # Layer 4: Variables for hidden weights and biases
# W_hidden_4 = tf.Variable(weight_initializer([n_neurons_3, n_neurons_4]))
# bias_hidden_4 = tf.Variable(bias_initializer([n_neurons_4]))

# out = tf.transpose(tf.add(tf.matmul(hidden_4, W_out), bias_out))

# Cost function
# mse = tf.reduce_mean(tf.squared_difference(out, Y))

# Optimizer
# opt = tf.train.AdamOptimizer().minimize(mse)

# Minibatch training
#     for i in range(0, len(y_train) // batch_size):
#         start = i * batch_size
#         batch_x = X_train[start:start + batch_size]
#         batch_y = y_train[start:start + batch_size]
#         # Run optimizer with batch
#         net.run(opt, feed_dict={X: batch_x, Y: batch_y})

#         # Show progress
#         if np.mod(i, 5) == 0:
#             # Prediction
#             pred = net.run(out, feed_dict={X: X_test})
#             line2.set_ydata(pred)
#             plt.title('Epoch ' + str(e) + ', Batch ' + str(i))
#             file_name = 'img/epoch_' + str(e) + '_batch_' + str(i) + '.jpg'
#             plt.savefig(file_name)
#             plt.pause(0.01)
# Print final MSE after Training
# mse_final = net.run(mse, feed_dict={X: X_test, Y: y_test}) 

