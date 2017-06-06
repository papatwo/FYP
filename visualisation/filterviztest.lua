require 'nn'
require 'image'
local image = require 'image'
local loadcaffe = require 'loadcaffe'
require 'dpnn'
require 'TVCriterion'

--local prototxt = '/home/akaishuushan/FYP/illu2vec/illust2vec.prototxt' -- define prototxt path
--local caffemodel='/home/akaishuushan/FYP/illu2vec/illust2vec_ver200.caffemodel' -- define caffemodel path
 prototxt = '/home/akaishuushan/FYP/VGG_CNN_S/VGG_CNN_S_deploy.prototxt' -- define prototxt path
caffemodel='/home/akaishuushan/FYP/VGG_CNN_S/VGG_CNN_S.caffemodel' -- define caffemodel path
vgg = loadcaffe.load(prototxt, caffemodel, 'nn') 

for l = 1, 13 do --remove layer from bottom to top: output
  vgg:remove()                                                             --|
end 

--[[[input -> (1) -> (2) -> (3) -> (4) -> (5) -> (6) -> (7) -> (8) -> (9) -> (10) -> output]
  (1): nn.SpatialConvolution(3 -> 96, 7x7, 2,2)
  (2): nn.ReLU
  (3): nn.SpatialCrossMapLRN
  (4): nn.SpatialMaxPooling(3x3, 3,3)
  (5): nn.SpatialConvolution(96 -> 256, 5x5)
  ----------------------------------------------------------
  (6): nn.ReLU
  (7): nn.SpatialMaxPooling(2x2, 2,2)
  (8): nn.SpatialConvolution(256 -> 512, 3x3, 1,1, 1,1)
  (9): nn.ReLU
  (10): nn.SpatialConvolution(512 -> 512, 3x3, 1,1, 1,1)
]]

img = image.load('golden.jpg')
gradLoss = torch.zeros(vgg:forward(img):size())
filteridx = 256
gradLoss[filteridx] = 1
print(gradLoss)
for i = 1, 20 do
-- eg. remore 18 layers, output from conv layer 5
out = vgg:forward(img) --size 256x26x41
loss = torch.mean(out[filteridx])
-- compute the gradient of the input picture wrt this loss
grads = vgg:backward(img, gradLoss) 
-- we normalize the gradient
grads:div(math.sqrt(torch.pow(grads, 2):mean()) + 1e-5)
img:add(grads)
image.display(img)
end
--g = vgg:updateGradInput(img, dst) 
--o =  vgg:get(1).output
--w = vgg.modules[1].weight --the filter in the first convolutional layer
--image.display(w)
--[[for i=1,w:size()[1] do
image.display(w[i])
end]]


    --image.display(o)
