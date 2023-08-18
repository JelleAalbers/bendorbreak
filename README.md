bendorbreak
============

Robustness tests for inference on strong gravitational lenses with neural networks.

Features:
  * **quicktest**: one command to simulate images with an altered config, then run a network and hierarchical inference on them.
  * **Saliency maps** using integrated gradients from [alibi](https://docs.seldon.io/projects/alibi/en/stable/examples/integrated_gradients_imagenet.html). See which pixels a prediction is most sensitive to.

Related packages:
  * [paltas](https://github.com/swagnercarena/paltas): Package to simulate lenses, train networks in TensorFlow, and perform hierarchical inference on the network predictions.
  * [deepdarksub](https://github.com/JelleAalbers/deepdarksub): earlier attempt to wrap paltas and train networks with FastAI.
