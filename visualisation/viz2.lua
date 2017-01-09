local image = require 'image'
local loadcaffe = require 'loadcaffe'
require 'dpnn'
require 'TVCriterion'

-- Options
local imgSize = 224 -- Can be smaller than 224 x 224 input as only using convolutional layers
local filterIndex = 1 -- Must index a valid convolutional filter

-- Load VGG (small)
local vgg = loadcaffe.load('VGG_CNN_S_deploy.prototxt', 'VGG_CNN_S.caffemodel', 'nn')
-- Remove fully connected layers
for l = 1, 11 do
  vgg:remove()
end
--print(vgg:get(vgg:size()))

-- Construct preprocessing network
local model = nn.Sequential()
--model:add(nn.TVCriterion(0))
model:add(nn.MulConstant(255)) -- Do not perform inplace
model:add(nn.SplitTable(1, 3))

local shuffleModel = nn.ConcatTable()

-- Technically the mean image should be subtracted, but the mean channels are pretty close
local BModel = nn.Sequential()
BModel:add(nn.SelectTable(3)) -- Select B channel
BModel:add(nn.View(-1, 1, imgSize, imgSize)) -- Add back channel dimension
BModel:add(nn.AddConstant(-103.939, true)) -- Subtract B mean pixel value

local GModel = nn.Sequential()
GModel:add(nn.SelectTable(2)) -- Select G channel
GModel:add(nn.View(-1, 1, imgSize, imgSize)) -- Add back channel dimension
GModel:add(nn.AddConstant(-116.779, true)) -- Subtract G mean pixel value

local RModel = nn.Sequential()
RModel:add(nn.SelectTable(2)) -- Select R channel
RModel:add(nn.View(-1, 1, imgSize, imgSize)) -- Add back channel dimension
RModel:add(nn.AddConstant(-123.68, true)) -- Subtract R mean pixel value

shuffleModel:add(BModel)
shuffleModel:add(GModel)
shuffleModel:add(RModel)
model:add(shuffleModel) -- RGB -> BGR shuffle
model:add(nn.JoinTable(1, 3))
model:add(nn.Squeeze()) -- Remove singleton dimension
model:add(vgg)

-- Create white noise image
local img = torch.Tensor(3, imgSize, imgSize):uniform(-1e-3, 1e-3):add(0.5)
-- Create target gradient to maximise mean of filter
local gradLoss = torch.zeros(model:forward(img):size()) -- Work out size automatically from input and model
gradLoss[filterIndex] = 1 -- Visualise one filter
-- Display
local win = image.display({image = img})

-- Maximise mean activation of filter
for i = 1, 50 do
  local output = model:forward(img)
  local loss = torch.mean(output[filterIndex]) -- Mean activation of filter (which should be maximised)
  print(loss)
  local imgLoss = model:backward(img, gradLoss)
  imgLoss:div(math.sqrt(torch.pow(imgLoss, 2):mean()) + 1e-5) -- Normalise gradient

  img:add(imgLoss) -- No range normalisation within loop
  image.display({image = img, win = win})
end