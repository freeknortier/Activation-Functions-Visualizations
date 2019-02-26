import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plot

#SoftPlus
x = np.linspace(-10, 10, 50)
output = tf.nn.softplus(x)
initialization = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(initialization)
    y = session.run(output)
    plot.xlabel('Neuron Activity')
    plot.ylabel('Neuron Output')
    plot.title('SoftPlus Activation Function')
    plot.plot(x, y)
    plot.show()
