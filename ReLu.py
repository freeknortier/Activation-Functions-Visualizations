import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plot

#ReLu
x = np.linspace(-10, 10, 50)
output = tf.nn.relu(x)
initialization = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(initialization)
    y = session.run(output)
    plot.xlabel('Neuron Activity')
    plot.ylabel('Neuron Output')
    plot.title('ReLu Activation Function')
    plot.plot(x, y)
    plot.show()

