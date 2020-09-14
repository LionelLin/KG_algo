import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import activations
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers

Dropout = keras.layers.Dropout
LeakyReLU = keras.layers.LeakyReLU
Softmax = keras.layers.Softmax
Concatenate = tf.keras.layers.concatenate
Average = tf.keras.layers.average


class GraphAttentionLayer(keras.layers.Layer):
    def __init__(self,
                 attn_head,
                 output_dim,

                 attn_heads_reduction='concat',
                 activation='relu',

                 dropout_rate=None,

                 kernel_initializer=None,
                 bias_initializer=None,
                 attn_kernel_initializer=None,

                 kernel_regularizer=None,
                 bias_regularizer=None,
                 attn_kernel_regularizer=None,
                 activity_regularizer=None,

                 kernel_constraint=None,
                 bias_constraint=None,
                 attn_kernel_constraint=None,
                 **kwargs):
        super(GraphAttentionLayer, self).__init__(**kwargs)

        # 最后有多少个"头"的attention
        self.attn_head = attn_head

        # 选择拼接的方式还是平均的方式
        self.attn_heads_reduction = attn_heads_reduction

        # 原论文中的F'
        self.output_dim = output_dim

        # dropout rate
        self.dropout_rate = dropout_rate

        # 初始化器
        self.kernel_initializer = initializers.get(kernel_initializer),
        self.bias_initializer = initializers.get(bias_initializer),
        self.attn_kernel_initializer = initializers.get(attn_kernel_initializer),

        # 正则化器
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.attn_kernel_regularizer = regularizers.get(attn_kernel_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        # 约束
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.attn_kernel_constraint = constraints.get(attn_kernel_constraint)

        # 激活函数
        self.activation = activations.get(activation)

        # weight matrix， 原论文中的W
        self.kernels = []

        # 原论文中的a[:F']
        self.attn_kernels_self = []

        # 原论文中的a[F':]
        self.attn_kernels_neighbors = []

    def build(self, input_shape):
        assert input_shape >= 2
        input_dim = input_shape[0][-1]
        for head in range(self.attn_head):
            kernel = self.add_weight(shape=(input_dim, self.output_dim),
                                     initializer=self.kernel_initializer,
                                     regularizer=self.kernel_regularizer,
                                     constraint=self.kernel_constraint,
                                     name='kernel_{}'.format(head))
            self.kernels.append(kernel)

            # 样本本身的kernel初始化
            attn_kernel_self = self.add_weight(shape=(self.output_dim, 1),
                                               initializer=self.attn_kernel_initializer,
                                               regularizer=self.attn_kernel_regularizer,
                                               constraint=self.attn_kernel_constraint,
                                               name='attn_kernel_self_{}'.format(head))
            self.attn_kernels_self.append(attn_kernel_self)

            # 样本邻居的kernel初始化
            attn_kernel_neighbors = self.add_weight(shape=(self.output_dim, 1),
                                                    initializer=self.attn_kernel_initializer,
                                                    regularizer=self.attn_kernel_regularizer,
                                                    constraint=self.attn_kernel_constraint,
                                                    name='attn_kernel_neighbors_{}'.format(head))
            self.attn_kernels_neighbors.append(attn_kernel_neighbors)
        self.built = True

    def call(self, inputs, training=True):
        """
        :param inputs: 两个tensor, 第一个表示样本本身（self），维度为1*N；第二个tensor表示它的邻居，
                       维度为nerighbor_num*N
        :param training:
        :return:
        """
        self_X_origin = inputs[0]  # 1 * F
        neighbor_X_origin = inputs[1]  # N *F

        outputs_vec_list = []
        for head in range(self.attn_heads):
            kernel = self.kernels[head]  # F * F'

            self_X = tf.matmul(self_X_origin, kernel)  # 1 * F'
            neighbor_X = tf.matmul(neighbor_X_origin, kernel)  # N * F'

            attn_kernel_self = self.attn_kernels_self[head]  # F' * 1
            attn_kernel_neighbor = self.attn_kernels_neighbors[head]  # F' * 1

            # a^T * [self_fea, neighbor_fea] = a_self^T*self_fea + a_neighbor*neighbor_fea
            attn_for_self = tf.matmul(self_X, attn_kernel_self)  # 1 * 1
            attn_for_neighbor = tf.matmul(neighbor_X, attn_kernel_neighbor)  # N * 1

            attn_for_combine = attn_for_self + attn_for_neighbor  # N * 1

            attn_for_combine = LeakyReLU(alpha=0.2)(attn_for_combine)  # N * 1

            attn_for_combine = Softmax(attn_for_combine)  # N * 1

            dropout_combine = Dropout(self.dropout_rate)(attn_for_combine)  # N * 1

            output_self_X = Dropout(self.dropout_rate)(self_X)  # 1 * F'

            output_neighbor_X = tf.matmul(tf.transpose(dropout_combine), neighbor_X)  # 1 * F'

            output_combine = output_self_X + output_neighbor_X

            outputs_vec_list.append(output_combine)

        if self.attn_heads_reduction == 'concat':
            output = Concatenate(outputs_vec_list, name='attn_output_concat', axis=1)
        else:
            output = Average(outputs_vec_list, name='attn_output_avg', axis=0)

        return self.attention(output)

    def compute_output_shape(self, input_shape):
        if self.attn_heads_reduction == 'concat':
            output_vec_dim = self.output_dim * self.attn_head
        else:
            output_vec_dim = self.output_dim

        output_shape = 1, output_vec_dim
        return output_shape
