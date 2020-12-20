# Aegis projection node

Projects one shape into another with a fully connected and optionally squashed neural network layer.
This node behaves as a proxy, so when calling it, it'll fetch the actual data from another node first.

## Environment
* `PORT` - port to listen on
* `SOURCE_URL` - the source data to project
* `INPUT_SHAPE` - the input shape as a whitespace separated list
* `OUTPUT_SHAPE` - the target output shape as a whitespace separated list
* `SQUASH` - Boolean; whether to squash (apply tanh) to the output or not
* `DENSITY` - density of the weight array, defaults to 0.1 (or 10%)
* `PRE_SCALE` - amount to scale pre-squashed values by

## TODO
* Saving/loading
* `POST_SCALE`
* Add values for scaling and centering after squash
