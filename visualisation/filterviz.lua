require 'nn'
require 'image'
local image = require 'image'
local loadcaffe = require 'loadcaffe'
require 'dpnn'
require 'TVCriterion'
require 'preprocess'
require 'reducenet'

-- Options
local imgSize = 224 -- Can be smaller than 224 x 224 input as only using convolutional layers
local filterIndex = 80 -- Must index a valid convolutional filter

-- Load VGG (small)
local prototxt = '/home/akaishuushan/FYP/VGG_CNN_S/VGG_CNN_S_deploy.prototxt' -- define prototxt path
local caffemodel='/home/akaishuushan/FYP/VGG_CNN_S/VGG_CNN_S.caffemodel' -- define caffemodel path
local vgg = loadcaffe.load(prototxt, caffemodel, 'nn') 
local img = image.load('golden.jpg')
net = reducenet(vgg, 15) -- input: full net and end layer
-- Construct preprocessing network
local model = nn.Sequential()
--model:add(nn.TVCriterion(0))
model:add(net)

-- Create white noise image
--local img = torch.Tensor(3, imgSize, imgSize):uniform(-1, 1):add(100)
local img = torch.Tensor(3, imgSize, imgSize):uniform(-1, 1):mul(20):add(128)
-- Create target gradient to maximise mean of filter
local gradLoss = torch.zeros(model:forward(img):size()) -- Work out size automatically from input and model
gradLoss[filterIndex] = 1 -- Visualise one filter (all other  filter indices are 0 but the selected filter is 1)
-- Display
local win = image.display({image = img})