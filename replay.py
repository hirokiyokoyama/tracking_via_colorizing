import tensorflow as tf

def stratified_sample(probs, n):
    N = tf.shape(probs)[0:1]
    c = tf.cumsum(probs)
    c = c/c[-1]
    borders = tf.linspace(0.,1.,n+1)
    right = borders[1:]

    c = tf.expand_dims(c, 0)
    right = tf.expand_dims(right, 1)
    greater_mask = tf.cast(tf.greater(c, right), tf.int32)
    _cum_num = tf.reduce_sum(greater_mask, 1)
    cum_num = tf.concat([N,_cum_num[:-1]],0)
    num = cum_num - _cum_num
    unif = tf.contrib.distributions.Uniform(low=0., high=tf.cast(num, tf.float32))
    local_inds = tf.cast(unif.sample(), tf.int32)
    begin = N - cum_num
    return local_inds + begin

class PrioritizedHistory:
    def __init__(self, name_to_shape_dtype,
                 capacity = 100000,
                 device='/gpu:0',
                 variable_collections=['history']):
        variables = []
        self._capacity = capacity
        self._device = device
        
        with tf.device(self._device):
            self._histories = {}
            for name, (shape, dtype) in name_to_shape_dtype.iteritems():
                self._histories[name] = tf.Variable(tf.zeros([capacity]+list(shape), dtype=dtype),
                                                trainable = False, collections=variable_collections)
                variables.append(self._histories[name])
        
            self._weights = tf.Variable(tf.zeros([capacity], dtype=tf.float32),
                                        trainable = False, collections=variable_collections)
            variables.append(self._weights)

            self._inds = tf.Variable(tf.range(capacity),
                                     trainable = False, collections=variable_collections)
            variables.append(self._inds)
            
            self._size = tf.Variable(tf.constant(0, dtype=tf.int32),
                                     trainable = False, collections=variable_collections)
            variables.append(self._size)
        
            self.saver = tf.train.Saver(var_list=variables)
            self.initializer = tf.group(map(lambda v: v.initializer, variables))

    def append(self, name_to_value, weight):
        with tf.device(self._device):
            weight = tf.convert_to_tensor(weight)
            name_to_value = {name: tf.convert_to_tensor(value) for name, value in name_to_value.iteritems()}
            inds = tf.where(tf.less(self._weights, weight))
            accepted = tf.greater(tf.shape(inds)[0], 0)
            def insert():
                ind = inds[0,0]
                ind_to_be_replaced = self._inds[-1]
                ops = []
                for name, value in name_to_value.iteritems():
                    ops.append(self._histories[name][ind_to_be_replaced].assign(value))
                with tf.control_dependencies(ops):
                    ops = [self._weights[(ind+1):].assign(self._weights[ind:-1]),
                           self._inds[(ind+1):].assign(self._inds[ind:-1])]
                    with tf.control_dependencies(ops):
                        ops = [self._weights[ind].assign(weight),
                               self._inds[ind].assign(ind_to_be_replaced),
                               self._size.assign(tf.reduce_min([self._size+1, self._capacity]))]
                        with tf.control_dependencies(ops):
                            return tf.cast(ind, tf.int32)
            return tf.cond(accepted, insert, lambda: -1)

    def update_weight(self, ind, weight):
        with tf.device(self._device):
            ind = tf.convert_to_tensor(ind)
            old_weight = self._weights[ind]
            ind_to_be_moved = self._inds[ind]
            weight = tf.convert_to_tensor(weight)
            def first_less():
                inds = tf.where(tf.less(self._weights, weight))
                return tf.cond(tf.greater(tf.shape(inds)[0], 0),
                               lambda: tf.cast(inds[0,0], tf.int32),
                               lambda: tf.constant(self._capacity-1, dtype=tf.int32))
            def last_greater():
                inds = tf.where(tf.greater(self._weights, weight))
                return tf.cond(tf.greater(tf.shape(inds)[0], 0),
                               lambda: tf.cast(inds[-1,0], tf.int32),
                               lambda: tf.constant(self._capacity-1, dtype=tf.int32))
            new_ind = tf.cond(tf.greater(weight, old_weight), first_less, last_greater)
            def up():
                ops = [self._weights[ind:new_ind].assign(self._weights[(ind+1):(new_ind+1)]),
                       self._inds[ind:new_ind].assign(self._inds[(ind+1):(new_ind+1)])]
                return tf.group(ops)
            def down():
                ops = [self._weights[(new_ind+1):(ind+1)].assign(self._weights[new_ind:ind]),
                       self._inds[(new_ind+1):(ind+1)].assign(self._inds[new_ind:ind]),]
                return tf.group(ops)
            with tf.control_dependencies([ind_to_be_moved]):
                shift = tf.cond(tf.greater(new_ind, ind), up, down)
            with tf.control_dependencies([shift]):
                ops = [self._weights[new_ind].assign(weight),
                       self._inds[new_ind].assign(ind_to_be_moved)]
                with tf.control_dependencies(ops):
                    return tf.identity(new_ind)
    
    def sample(self, size):
        with tf.device(self._device):
            inds = stratified_sample(self._weights[:self._size], size)
            inds = tf.gather(self._inds, inds)
            return inds, {name: tf.gather(hist, inds) for name, hist in self._histories.iteritems()}
