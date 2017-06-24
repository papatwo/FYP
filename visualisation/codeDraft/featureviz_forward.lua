local image = require 'image'
local loadcaffe = require 'loadcaffe'
require 'dpnn'
require 'TVCriterion'
require 'reducenet'

-- Load VGG (small)
local prototxt = '/home/akaishuushan/FYP/VGG_CNN_S/VGG_CNN_S_deploy.prototxt' -- define prototxt path
local caffemodel = '/home/akaishuushan/FYP/VGG_CNN_S/VGG_CNN_S.caffemodel' -- define caffemodel path
vgg = loadcaffe.load(prototxt, caffemodel, 'nn') 
net = reducenet(vgg, 8) -- input: full net and end layer
--local img= torch.rand(3,224,224):uniform()
local img = image.load('golden.jpg')

neuron = net:get(1).weight
--neuron = net.mdoules[2?1?].weight
output = net:forward(img)
neuron_out = net:get(1).output
--params, gradParams = model:parameters() -- get the params for each layer even in the table layer



