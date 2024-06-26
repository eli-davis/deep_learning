# B"H

 For additional info, check out https://www.youtube.com/watch?v=gJ5UENsAQGY

Here's a couple more notes:
- Convolution comes from "convolution" in signal processing, where a portion of a signal is amplified or attenuated
- Basically each layer is N 2D channels
- So first layer could be 32 2D channels
  Each channel is in 2D image space
  If input image is 960x1280x3
  First layer could be 480x640x32
    + Where the first layer is activations based on a filter (2D weights of 3x3 that filter on the original image, with a stride x2 that leads to the downsample)
    + The 2D 3x3 filter is finding edges and color gradients primarily
    + Then the first layer is 32 "activation maps" of 640x480 - each activation in the map corresponds to whether the location in the original image "activates" for each particular filter
    + So channel 1 could be looking for right to left edges – with layer 1 activating anywhere this occurs in the original image
    + ... So on for channels 2-32, each is a learned 3x3 filter that turns into an activation map

Especially 23:00 in to the video this is described in detail.

Also the deep visualization toolbox @ https://www.youtube.com/watch?v=AgkfIQ4IGaM shows this happening to build an intution
