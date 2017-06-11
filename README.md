# bio-intelligent-hw
# This branch is "frozen" as of June 12th, 2017.

Homework for "Bio Intelligent Algorithms".


Homework is to train deep neural network (no convolution allowed, only fully connected) on MNIST dataset.

Test different settings (learning rate, optimizer etc.), different data augmentation (noise, rotation etc) and reach ~95% success with KNN (e.g. good dimension reduction). Plot clustering using tSNE.

Project was done using TensowFlow.

Results:
95.9% accuracy with KNN (k=1) on 10k test set.


Several notes / conclusions / TODOs:
- Activation functions issue
-- ReLu vs sigmoid - for some reason loss was exploding with ReLu.
-- TODO: test tanh activation as well
- L2 vs L1 - when to use each?
-- Add L1 penalty to testing
- Add error on validation
- Add noise
- global vs local normalization
    currently using per image normalization
- Autoencoders and dropout layers
- We want the algorithm to reproduce dropped-out parts of the image
- looks like comparing to ALL nearest neighbors actually improves match %
-- Better results when searched nearest neighbor in 10k images graph than 2k images graph
- tSNE: better implementation (Barnes-Hut); also search for better perplexity
- Add train loss (currently minibatch) every DISPLAY_STEP_NORMAL steps
- Add smarter parameter search, using sigopt.com or other Bayesian search methods for example.
