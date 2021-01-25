import tensorflow as tf

class PhaseSplitter(tf.keras.layers.Layer):

   def call(self, inputs):
       
       data, phase = inputs
       phase = tf.reshape(phase,[-1])
       asc_data = tf.boolean_mask(data, tf.equal(phase, 1))
       cru_data = tf.boolean_mask(data, tf.equal(phase, 2))
       des_data = tf.boolean_mask(data, tf.equal(phase, 3))
       
       asc_index = tf.where(tf.equal(phase,1))
       cru_index = tf.where(tf.equal(phase,2))
       des_index = tf.where(tf.equal(phase,3))
       
       concat_index = tf.reshape(tf.concat([asc_index,
                                            cru_index,
                                            des_index], 0),[-1])
       
       return asc_data, cru_data, des_data, concat_index
   
   def get_config(self):
       config = super(PhaseSplitter, self).get_config()
       config.update({"trainable": False})
       return config
