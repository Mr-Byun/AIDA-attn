from tensorflow import transpose, matmul
from tensorflow.nn import dropout, softmax
from tensorflow.math import multiply
from tensorflow.keras import layers, initializers, regularizers, activations
from tensorflow.keras import backend as K

class SpatialAttention(layers.Layer):
    def __init__(self,
                 units=32,
                 dropout=0.2,
                 return_attention=False,
                 activation='tanh',
                 kernel_initializer='glorot_normal',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 **kwargs):
        super(SpatialAttention, self).__init__(**kwargs)
        self.units = units
        self.dropout = dropout
        self.return_attention=return_attention
        
        self.activation = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.Wh, self.bh = None, None
        self.Wa, self.ba = None, None
        
    def build(self, input_shape):
        time_dim = int(input_shape[1])
        self.Wh = self.add_weight(shape=(time_dim, self.units),
                                  name='{}_Add_Wh'.format(self.name),
                                  initializer=self.kernel_initializer,
                                  regularizer=self.kernel_regularizer)
        self.bh = self.add_weight(shape=(self.units,),
                                  name='{}_Add_bh'.format(self.name),
                                  initializer=self.bias_initializer,
                                  regularizer=self.bias_regularizer)

        self.Wa = self.add_weight(shape=(self.units, time_dim),
                                  name='{}_Add_Wa'.format(self.name),
                                  initializer=self.kernel_initializer,
                                  regularizer=self.kernel_regularizer)
        self.ba = self.add_weight(shape=(time_dim,),
                                  name='{}_Add_ba'.format(self.name),
                                  initializer=self.bias_initializer,
                                  regularizer=self.bias_regularizer)
        super(SpatialAttention, self).build(input_shape)
    
    def call(self, inputs, **kwargs):
        inputs_T = transpose(inputs, perm=[0, 2, 1])
        h = matmul(inputs_T, self.Wh) + self.bh
        h = self.activation(h)
        h = dropout(h, self.dropout)
        
        a = matmul(h, self.Wa) + self.ba
        a_probs = transpose(softmax(a), perm=[0, 2, 1])
        
        spatial_emb = multiply(inputs, a_probs)
        if self.return_attention:
            return [spatial_emb, a_probs]
        return spatial_emb



class TemporalAttention(layers.Layer):
    def __init__(self,
                 units=32,
                 return_attention=False,
                 kernel_initializer='glorot_normal',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 **kwargs):
        super(TemporalAttention, self).__init__(**kwargs)
        self.units = units
        self.return_attention = return_attention
        self.attention_width = int(1e9)
        
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.Wx, self.Wt, self.bh = None, None, None
        self.Wa, self.ba = None, None

    def build(self, input_shape):
        feature_dim = int(input_shape[2])
        self.Wt = self.add_weight(shape=(feature_dim, self.units),
                                  name='{}_Add_Wt'.format(self.name),
                                  initializer=self.kernel_initializer,
                                  regularizer=self.kernel_regularizer)
        self.Wx = self.add_weight(shape=(feature_dim, self.units),
                                  name='{}_Add_Wx'.format(self.name),
                                  initializer=self.kernel_initializer,
                                  regularizer=self.kernel_regularizer)
        self.bh = self.add_weight(shape=(self.units,),
                                  name='{}_Add_bh'.format(self.name),
                                  initializer=self.bias_initializer,
                                  regularizer=self.bias_regularizer)

        self.Wa = self.add_weight(shape=(self.units, 1),
                                  name='{}_Add_Wa'.format(self.name),
                                  initializer=self.kernel_initializer,
                                  regularizer=self.kernel_regularizer)
        self.ba = self.add_weight(shape=(1,),
                                  name='{}_Add_ba'.format(self.name),
                                  initializer=self.bias_initializer,
                                  regularizer=self.bias_regularizer)
        super(TemporalAttention, self).build(input_shape)

    def call(self, inputs, **kwargs):
        input_len = K.shape(inputs)[1]
        input_shape = K.shape(inputs)
        batch_size, input_len = input_shape[0], input_shape[1]

        # h_{t, t'} = \tanh(x_t^T W_t + x_{t'}^T W_x + b_h)
        q = K.expand_dims(K.dot(inputs, self.Wt), 2)
        k = K.expand_dims(K.dot(inputs, self.Wx), 1)
        h = K.tanh(q + k + self.bh)
        
        # e_{t, t'} = W_a h_{t, t'} + b_a
        e = K.reshape(K.dot(h, self.Wa) + self.ba, (batch_size, input_len, input_len))
        
        lower = K.arange(0, input_len) - (self.attention_width - 1) # causal
        lower = K.expand_dims(lower, axis=-1)
        upper = lower + self.attention_width
        indices = K.expand_dims(K.arange(0, input_len), axis=0)
        e -= 10000.0 * (1.0 - K.cast(lower <= indices, K.floatx()) * K.cast(indices < upper, K.floatx()))

        # a_{t} = \text{softmax}(e_t)
        e = K.exp(e - K.max(e, axis=-1, keepdims=True))
        a = e / K.sum(e, axis=-1, keepdims=True)

        # l_t = \sum_{t'} a_{t, t'} x_{t'}
        v = K.batch_dot(a, inputs)

        if self.return_attention:
            return [v, a]
        return v
    
