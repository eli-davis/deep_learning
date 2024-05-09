recommend looking into the calculus method of increments (morris kline chapter 1-2 calculus if you can find it)
refreshes idea that derivative is a rate_of_change of one value with respect to another
e.g. position_change relative to time ("speed")
at an instantaneous value
method of increments calculate this rate manually over successively smaller intervals, to get an approximation of the rate of change at an "instant"

look into weight initialization (0-1 normal distribution currently) OK
more detaiil on weight initialization
generally good references: justin johnson umich videos on youtube + andrej karpath cs231n notes
at 598_WI2022_lecture09.pdf

softmax
https://github.com/eli-davis/deep_learning/blob/main/assignment1/softmax.png

look into why vs naive implementaion of output1 / (output1+...+output9)
derivative smooth/outputs sum to1
cross entropy
more detailed why, justin johnson videos / andrej karpathy stanford cs231n notes will help here as well

overall weight initialization, activations like softmax, loss funcitons like cross entropt, how to do gradients/weight updates (we'll get to back propogation are the real core ideas in neural networks
gradients and weight updates are two sides of the same coin
gradient is how much does changing this neuron affect the output loss of the whole network
based on the gradient (of error with_respect_to_loss) we update the weight proportionally
if you have positive error_gradient it means if we increase this weight, the error_of_the_overall_models_output_predictions will increase
if we have negatiber error_gradient it means if we decrease this weight, the error_of_the_overall_models_output_predictions will decrease
so, basically increase weights when you have a negative error gradient (and vice versa)
networks get deeper, these fundamentals are the same, with implications based on the network architechture

back prop
regardless of number of parameters, we want the error_gradient_of_the_overall_models_output_predictions with respect to this specific weight
in a 10 layer network, first we compute the error gradient of the outputs/layer10
then we compute layer9 weight error_gradients (which is a function of layer10 error gradients)
L1 is f(L2 which is f(3 which if f(L4.... etc

next step: for each training_iteration ("referred to as epoch")
for each image:
image (dot) weights = predictions
error_gradient is function of ( predictions, expected output)
update weights based on error gradient ("error gradient is derivative_of_ of each weight with respect to output predictions ... i.e. if we make a small increment to this weight does the overall error go up or down)
at each iteration:
once youve updated the weights after going thru each image
you want a measure of how well its working
run the validation images, get the predicted digit 0-9 (i.e. max probability of the outputs) and print out what percent of the validation images were correct

finally, 
run the test images, get the predicted digit 0-9 (i.e. max probability of the outputs) and print out what percent of the validation images were correct

overall: try to make changes so that your validation and test results are as good as possible, once done visualize the weight a 16x16 images to see what they look like
