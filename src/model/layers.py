
import tensorflow as tf
import numpy as np


class TrainableRescaling(tf.keras.layers.Layer):
    def __init__(self, input_dim, trainable_obs_weight):
        super(TrainableRescaling, self).__init__()
        self.input_dim = input_dim
        self.trainable_obs_weight = trainable_obs_weight
        self.w = tf.Variable(
            initial_value=tf.ones(shape=(1, input_dim)), dtype=tf.dtypes.float32,
            trainable=self.trainable_obs_weight, name="trainable_rescaling")

    def call(self, inputs):
        return tf.math.multiply(
            inputs,
            tf.repeat(self.w, repeats=tf.shape(inputs)[0], axis=0))


class ScalableDataReuploadingController(tf.keras.layers.Layer):
    def __init__(
            self,
            num_input_params,
            num_params,
            circuit_depth,
            params,
            trainable_scaling=True,
            use_reuploading=True,
            name="scalable_data_reuploading"):
        super(ScalableDataReuploadingController, self).__init__(name=name)
        self.num_params = num_params
        self.circuit_depth = circuit_depth
        self.use_reuploading = use_reuploading

        self.num_input_params = num_input_params
        if self.use_reuploading:
            self.num_input_params *= circuit_depth

        param_init = tf.random_uniform_initializer(minval=0., maxval=np.pi)
        self.params = tf.Variable(
            initial_value=param_init(shape=(1, num_params), dtype=tf.dtypes.float32),
            trainable=True, name="params"
        )

        input_param_init = tf.ones(shape=(1, self.num_input_params))
        self.input_params = tf.Variable(
            initial_value=input_param_init, dtype=tf.dtypes.float32,
            trainable=trainable_scaling, name="input_params"
        )

        alphabetical_params = sorted(params)
        self.indices = tf.constant([params.index(a) for a in alphabetical_params])

    def call(self, inputs):
        output = tf.repeat(self.params, repeats=tf.shape(inputs)[0], axis=0)

        input_repeats = self.circuit_depth if self.use_reuploading else 1
        repeat_inputs = tf.reshape(
            tf.repeat(inputs, repeats=input_repeats, axis=0),
            [tf.shape(inputs)[0], self.num_input_params])
        repeat_inp_weights = tf.repeat(
            self.input_params, repeats=tf.shape(inputs)[0], axis=0)

        data_values = tf.math.multiply(repeat_inputs, repeat_inp_weights)
        output = tf.concat(
            [
                data_values,
                output,
            ], 1)

        output = tf.gather(output, self.indices, axis=1)

        return output


class EquivariantLayer(tf.keras.layers.Layer):
    def __init__(
            self,
            num_input_params,
            n_vars,
            n_edges,
            circuit_depth,
            params,
            name="equivariant_layer"):
        super(EquivariantLayer, self).__init__(name=name)
        self.num_input_params = num_input_params * circuit_depth
        self.num_params = 2 * circuit_depth
        self.circuit_depth = circuit_depth

        param_init = tf.ones(shape=(1, self.num_params), dtype=tf.dtypes.float32)
        self.params = tf.Variable(
            initial_value=param_init,
            trainable=True, name="params"
        )

        self.param_repeats = []
        for layer in range(self.circuit_depth):
            self.param_repeats.append(n_vars)
            self.param_repeats.append(n_edges)

        alphabetical_params = sorted(params)
        self.indices = tf.constant([params.index(a) for a in alphabetical_params])

    def call(self, inputs):
        repeated_params = tf.repeat(self.params, repeats=self.param_repeats)

        repeat_inputs = tf.reshape(
            tf.repeat(inputs, repeats=self.circuit_depth, axis=0),
            [tf.shape(inputs)[0], self.num_input_params])

        data_values = tf.math.multiply(repeat_inputs, repeated_params)
        output = tf.gather(data_values, self.indices, axis=1)

        return output
