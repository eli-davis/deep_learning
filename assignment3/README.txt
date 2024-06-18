# B"H

Assignment 3

Background (Theory)
(1) https://github.com/yosinski/deep-visualization-toolbox
(2) https://cs231n.github.io/convolutional-networks/

Background (PyTorch)
- refactor + clean this code with detailed comments for each line https://github.com/pytorch/examples/blob/main/mnist/main.py
- update the network architechture to below
- comments should be there for every line of code thats not obvious
- really the above code is just a reference, should all be your own code

Network architechture:
(source: https://cs.stanford.edu/people/karpathy/convnetjs/demo/mnist.html)
layer_defs.push({type:'input', out_sx:24, out_sy:24, out_depth:1});
layer_defs.push({type:'conv', sx:5, filters:8, stride:1, pad:2, activation:'relu'});
layer_defs.push({type:'pool', sx:2, stride:2});
layer_defs.push({type:'conv', sx:5, filters:16, stride:1, pad:2, activation:'relu'});
layer_defs.push({type:'pool', sx:3, stride:3});
layer_defs.push({type:'softmax', num_classes:10});


