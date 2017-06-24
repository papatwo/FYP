--[[local image = require 'image'
local loadcaffe = require 'loadcaffe'
local optim = require 'optim'
require 'dpnn'
require 'nn' 
--require 'TVCriterion'


--Four preprocessing steps:
--1. resize img
--2. crop img at 224x224
--3. change colour channel from RBG --> BGR
--4. rescale to 0-255 and subtract mean


local model = nn.Sequential() -- define a container
local prototxt = '/home/akaishuushan/FYP/VGG_CNN_S/VGG_CNN_S_deploy.prototxt' -- define prototxt path
local caffemodel='/home/akaishuushan/FYP/VGG_CNN_S/VGG_CNN_S.caffemodel' -- define caffemodel path
--local vgg = loadcaffe.load(prototxt, caffemodel, 'nn') -- load caffe model

model:add(nn.MulConstant(255,true))
print(model)]] 
-- for model construction test

--use qlua -i filename.ext to exe script
require 'torch'
local image = require 'image'
local img = torch.Tensor(3, 224, 224):normal(0, 0.5)
image.display(img)

-- exit qlua: ctrl+D or os.exit()
-- execute .lua file in qlua use 'dofile' as well
