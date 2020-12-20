# Aegis projection node

Projects one shape into another with a fully connected and optionally squashed neural network layer.
This node behaves as a proxy, so when calling it, it'll fetch the actual data from another node first.

## Environment
* `PORT` - port to listen on
* `SOURCE_URL` - the source data to project
* `INPUT_SHAPE` - the input shape as a whitespace separated list
* `OUTPUT_SHAPE` - the target output shape as a whitespace separated list
* `ACTIVATION` - name of Keras-supported activation to use, defaults to `None`
* `DENSITY` - density of the weight array, defaults to 1.0 (100%)
* `INPUT_SCALE` - amount to scale input values by before feeding to NN
* `OUTPUT_SCALE` - amount to scale post-NN values by

## TODO
* Reimplement density functionality
