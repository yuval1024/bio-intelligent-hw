# bio-intelligent-hw

Homework for "Bio Intelligent Algorithms".

Homework is to train deep neural network (not convolutional, only fully connected) on MNIST dataset.

Test different settings (learning rate, optimizer etc.), different data augmentation (noise, rotation etc) and reach ~95% success with KNN (e.g. good clustering).

Project was done using TensowFlow.

Several notes / conclusions / TODOs:
- Activation functions issue
-- ReLu vs sigmoid - why is it exploding with ReLu?
    When switched activation function from Sigmoid to ReLu, gradients were exploding.
-- TODO: test tanh activation as well
- L2 vs L1 - when to use each?
-- Add L1 penalty to testing
- Add error on validation
- global vs local normalization
    currently using per image normalization
- Autoencoders and dropout layers
-- we want the algorithm to reproduce dropped-out parts of the image
- looks like comparing to ALL nearest neighbors actually improves match %
-- better results when searched nearest neighbor in 10k images graph than 2k images graph
- TODOs:
-- tSNE: bettrer implementation (Barnes-Hut); also search for better perplexity
-- add train loss (currently minibatch) every DISPLAY_STEP_NORMAL steps

