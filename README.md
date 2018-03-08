# Tensorflow-Sample

output sample:
C:\Users\bhargava\Anaconda3\envs\TensorFlowPyCharm\python.exe C:/Users/bhargava/PycharmProjects/pycharmTensor/custom_estimator.py
INFO:tensorflow:Using default config.
WARNING:tensorflow:Using temporary folder as model directory: C:\Users\bhargava\AppData\Local\Temp\tmpmey7bg3s
INFO:tensorflow:Using config: {'_model_dir': 'C:\\Users\\bhargava\\AppData\\Local\\Temp\\tmpmey7bg3s', '_tf_random_seed': None, '_save_summary_steps': 100, '_save_checkpoints_steps': None, '_save_checkpoints_secs': 600, '_session_config': None, '_keep_checkpoint_max': 5, '_keep_checkpoint_every_n_hours': 10000, '_log_step_count_steps': 100, '_service': None, '_cluster_spec': <tensorflow.python.training.server_lib.ClusterSpec object at 0x000002526DD519B0>, '_task_type': 'worker', '_task_id': 0, '_global_id_in_cluster': 0, '_master': '', '_evaluation_master': '', '_is_chief': True, '_num_ps_replicas': 0, '_num_worker_replicas': 1}
INFO:tensorflow:Calling model_fn.
INFO:tensorflow:Done calling model_fn.
INFO:tensorflow:Create CheckpointSaverHook.
INFO:tensorflow:Graph was finalized.
2018-03-07 23:19:37.112980: I C:\tf_jenkins\workspace\rel-win\M\windows\PY\36\tensorflow\core\platform\cpu_feature_guard.cc:140] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2
INFO:tensorflow:Running local_init_op.
INFO:tensorflow:Done running local_init_op.
INFO:tensorflow:Saving checkpoints for 1 into C:\Users\bhargava\AppData\Local\Temp\tmpmey7bg3s\model.ckpt.
INFO:tensorflow:loss = 1547872.1, step = 1
INFO:tensorflow:global_step/sec: 650.269
INFO:tensorflow:loss = 33148.387, step = 101 (0.154 sec)
INFO:tensorflow:global_step/sec: 897.734
INFO:tensorflow:loss = 16735.283, step = 201 (0.112 sec)
INFO:tensorflow:global_step/sec: 910.382
INFO:tensorflow:loss = 14029.104, step = 301 (0.109 sec)
INFO:tensorflow:global_step/sec: 880.592
INFO:tensorflow:loss = 0.0027144281, step = 401 (0.115 sec)
INFO:tensorflow:global_step/sec: 910.24
INFO:tensorflow:loss = 0.0013758058, step = 501 (0.110 sec)
INFO:tensorflow:global_step/sec: 903.826
INFO:tensorflow:loss = 0.0009016698, step = 601 (0.110 sec)
INFO:tensorflow:global_step/sec: 904.192
INFO:tensorflow:loss = 0.0006220038, step = 701 (0.111 sec)
INFO:tensorflow:global_step/sec: 886.451
INFO:tensorflow:loss = 0.00051601365, step = 801 (0.113 sec)
INFO:tensorflow:global_step/sec: 901.989
INFO:tensorflow:loss = 0.000488949, step = 901 (0.112 sec)
INFO:tensorflow:Saving checkpoints for 1000 into C:\Users\bhargava\AppData\Local\Temp\tmpmey7bg3s\model.ckpt.
INFO:tensorflow:Loss for final step: 0.00043979363.
INFO:tensorflow:Calling model_fn.
INFO:tensorflow:Done calling model_fn.
INFO:tensorflow:Starting evaluation at 2018-03-08-04:19:39
INFO:tensorflow:Graph was finalized.
INFO:tensorflow:Restoring parameters from C:\Users\bhargava\AppData\Local\Temp\tmpmey7bg3s\model.ckpt-1000
INFO:tensorflow:Running local_init_op.
INFO:tensorflow:Done running local_init_op.
INFO:tensorflow:Finished evaluation at 2018-03-08-04:19:39
INFO:tensorflow:Saving dict for global step 1000: accuracy = 1.0, accuracy_baseline = 0.61538464, auc = 0.9999998, auc_precision_recall = 0.9999999, average_loss = 4.227667e-06, global_step = 1000, label/mean = 0.61538464, loss = 5.4959673e-05, prediction/mean = 0.61538875

Test set accuracy: 1.000

<generator object Estimator.predict at 0x000002526DFA71A8>
INFO:tensorflow:Calling model_fn.
INFO:tensorflow:Done calling model_fn.
INFO:tensorflow:Graph was finalized.
INFO:tensorflow:Restoring parameters from C:\Users\bhargava\AppData\Local\Temp\tmpmey7bg3s\model.ckpt-1000
INFO:tensorflow:Running local_init_op.
INFO:tensorflow:Done running local_init_op.

Prediction is "denied" (100.0%), expected "denied"

Process finished with exit code 0
