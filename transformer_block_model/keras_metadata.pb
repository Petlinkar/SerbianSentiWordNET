
�root"_tf_keras_sequential*�{"name": "sequential_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": false, "class_name": "Sequential", "config": {"name": "sequential_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, null, 64]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}, {"class_name": "TransformerBlock", "config": {"name": "transformer_block", "trainable": true, "dtype": "float32", "embed_dim": 64, "num_heads": 4, "ff_dim": 128, "rate": 0.1}}]}, "shared_object_id": 2, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 64]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, null, 64]}, "is_graph_network": true, "full_save_spec": {"class_name": "__tuple__", "items": [[{"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, null, 64]}, "float32", "input_1"]}], {}]}, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, null, 64]}, "float32", "input_1"]}, "keras_version": "2.10.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, null, 64]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "shared_object_id": 0}, {"class_name": "TransformerBlock", "config": {"name": "transformer_block", "trainable": true, "dtype": "float32", "embed_dim": 64, "num_heads": 4, "ff_dim": 128, "rate": 0.1}, "shared_object_id": 1}]}}}2
�root.layer_with_weights-0"_tf_keras_layer*�{"name": "transformer_block", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "TransformerBlock", "config": {"name": "transformer_block", "trainable": true, "dtype": "float32", "embed_dim": 64, "num_heads": 4, "ff_dim": 128, "rate": 0.1}, "shared_object_id": 1, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, 64]}}2
�	root.layer_with_weights-0.att"_tf_keras_layer*�	{"name": "multi_head_attention", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "MultiHeadAttention", "config": {"name": "multi_head_attention", "trainable": true, "dtype": "float32", "num_heads": 4, "key_dim": 64, "value_dim": 64, "dropout": 0.0, "use_bias": true, "output_shape": null, "attention_axes": {"class_name": "__tuple__", "items": [1]}, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 4}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 5}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "query_shape": {"class_name": "TensorShape", "items": [null, null, 64]}, "key_shape": {"class_name": "TensorShape", "items": [null, null, 64]}, "value_shape": {"class_name": "TensorShape", "items": [null, null, 64]}}, "shared_object_id": 6, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, 64]}}2
�root.layer_with_weights-0.ffn"_tf_keras_sequential*�{"name": "sequential", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": false, "class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, null, 64]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_input"}}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 64, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "shared_object_id": 14, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, 64]}, "is_graph_network": true, "full_save_spec": {"class_name": "__tuple__", "items": [[{"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, null, 64]}, "float32", "dense_input"]}], {}]}, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, null, 64]}, "float32", "dense_input"]}, "keras_version": "2.10.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, null, 64]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_input"}, "shared_object_id": 7}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 8}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 9}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 10}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 64, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 11}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 12}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 13}]}}}2
�$root.layer_with_weights-0.layernorm1"_tf_keras_layer*�{"name": "layer_normalization", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "LayerNormalization", "config": {"name": "layer_normalization", "trainable": true, "dtype": "float32", "axis": [2], "epsilon": 1e-06, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 15}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 16}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "shared_object_id": 17, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, 64]}}2
�$root.layer_with_weights-0.layernorm2"_tf_keras_layer*�{"name": "layer_normalization_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "LayerNormalization", "config": {"name": "layer_normalization_1", "trainable": true, "dtype": "float32", "axis": [2], "epsilon": 1e-06, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 18}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 19}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "shared_object_id": 20, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, 64]}}2
�"root.layer_with_weights-0.dropout1"_tf_keras_layer*�{"name": "dropout", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "shared_object_id": 21, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, 64]}}2
�"root.layer_with_weights-0.dropout2"_tf_keras_layer*�{"name": "dropout_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "shared_object_id": 22, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, 64]}}2
�C*root.layer_with_weights-0.att._query_dense"_tf_keras_layer*�{"name": "query", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "EinsumDense", "config": {"name": "query", "trainable": true, "dtype": "float32", "output_shape": [null, 4, 64], "equation": "abc,cde->abde", "activation": "linear", "bias_axes": "de", "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 23}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 24}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 25, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, 64]}}2
�D(root.layer_with_weights-0.att._key_dense"_tf_keras_layer*�{"name": "key", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "EinsumDense", "config": {"name": "key", "trainable": true, "dtype": "float32", "output_shape": [null, 4, 64], "equation": "abc,cde->abde", "activation": "linear", "bias_axes": "de", "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 26}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 27}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 28, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, 64]}}2
�E*root.layer_with_weights-0.att._value_dense"_tf_keras_layer*�{"name": "value", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "EinsumDense", "config": {"name": "value", "trainable": true, "dtype": "float32", "output_shape": [null, 4, 64], "equation": "abc,cde->abde", "activation": "linear", "bias_axes": "de", "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 29}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 30}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 31, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, 64]}}2
�F&root.layer_with_weights-0.att._softmax"_tf_keras_layer*�{"name": "softmax", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "Softmax", "config": {"name": "softmax", "trainable": true, "dtype": "float32", "axis": {"class_name": "__tuple__", "items": [3]}}, "shared_object_id": 32, "build_input_shape": {"class_name": "TensorShape", "items": [null, 4, null, null]}}2
�G,root.layer_with_weights-0.att._dropout_layer"_tf_keras_layer*�{"name": "dropout", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.0, "noise_shape": null, "seed": null}, "shared_object_id": 33, "build_input_shape": {"class_name": "TensorShape", "items": [null, 4, null, null]}}2
�H+root.layer_with_weights-0.att._output_dense"_tf_keras_layer*�{"name": "attention_output", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "EinsumDense", "config": {"name": "attention_output", "trainable": true, "dtype": "float32", "output_shape": [null, 64], "equation": "abcd,cde->abe", "activation": "linear", "bias_axes": "e", "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 34}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 35}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 36, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, 4, 64]}}2
�I2root.layer_with_weights-0.ffn.layer_with_weights-0"_tf_keras_layer*�{"name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 8}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 9}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 10, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}, "shared_object_id": 37}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, 64]}}2
�J2root.layer_with_weights-0.ffn.layer_with_weights-1"_tf_keras_layer*�{"name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 64, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 11}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 12}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 13, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}, "shared_object_id": 38}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, 128]}}2