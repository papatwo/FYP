local image = require 'image'
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

--|------------------------------------------------------------------------------------------------------|
--|----------------------create and load exsisting caffemodel--------------------------|
--|------------------------------------------------------------------------------------------------------|
local model = nn.Sequential() -- define a container
local prototxt = '/home/akaishuushan/FYP/VGG_CNN_S/VGG_CNN_S_deploy.prototxt' -- define prototxt path
local caffemodel='/home/akaishuushan/FYP/VGG_CNN_S/VGG_CNN_S.caffemodel' -- define caffemodel path
local vgg = loadcaffe.load(prototxt, caffemodel, 'nn') -- load caffe model


--|------------------------------------------------------------------------------------------------------|
--|---------------------embedded preprocessing into model--------------------------|
--|------------------------------------------------------------------------------------------------------|
--model:add(nn.TVCriterion(0))


-- get the pixel back to 255 as torch displays in range 0-1
model:add(nn.MulConstant(255, true)) -- true - operation in-place without using extra state memory
model:add(nn.SplitTable(1, 3))  -- takes tensor as input and output tables. split each color channel into a table member

-- rescale the smallest side to 256
local rescale = function (img)
	local loadSize = {3,256,256} 
	if img:size()[2]--[[height]] < img:size()[3]--[[width]] then --[[resize height to 256]]
	-- image.scale(src, width, height)
	img = image.scale(img, img:size()[3]*loadSize[2]/img:size()[2], loadSize[3])
	else
	img = image.scale(img, loadSize[2], img:size()[2]*loadSize[3]/img:size()[3])
	end
	return img
end

model:add(rescale(img))
print(model)


-- create a container for colour channel recombination
local shuffleModel = nn.ConcatTable() 

-- build up submodels for each colour channel separately
local BModel = nn.Sequential() 
BModel:add(nn.SelectTable(3)) -- Select B channel --> extract out as a single tensor as in 224x224
BModel:add(nn.View(-1, 1, 224, 224)) -- Add back channel dimension as in 1x224x224
BModel:add(nn.AddConstant(103.939, true)) -- Subtract B mean pixel value

local GModel = nn.Sequential()
GModel:add(nn.SelectTable(2)) -- Select G channel
GModel:add(nn.View(-1, 1, 224, 224)) -- Add back channel dimension
GModel:add(nn.AddConstant(116.779, true)) -- Subtract G mean pixel value

local RModel = nn.Sequential()
RModel:add(nn.SelectTable(1)) -- Select R channel
RModel:add(nn.View(-1, 1, 224, 224)) -- Add back channel dimension
RModel:add(nn.AddConstant(123.68, true)) -- Subtract R mean pixel value

-- add colour channlel submodels together to form the final shuffle model
shuffleModel:add(BModel)
shuffleModel:add(GModel)
shuffleModel:add(RModel)
model:add(shuffleModel) -- RGB -> BGR shuffle
model:add(nn.JoinTable(1, 3)) -- 1 is in row direction join ((as outputs from above layer are three members for each colour channel respectively
model:add(vgg)


--|------------------------------------------------------------------------------------------------------|
--|-------------------------Loss function of representations-----------------------------|
--|------------------------------------------------------------------------------------------------------|
local criterion = nn.MSECriterion() -- minimize mean squared error


-- randomly get a white noise image
local img = torch.Tensor(3, 224, 224):normal(0, 0.5)
image.display(img)

-- load natural image and pass to the model
local targetImage = image.load('golden.jpg')
-- target rep is from the natural image
local target = model:forward(targetImage) 
--local target = torch.zeros(1000) -- for test model working purpose
--target[1] = 1

for i = 1, 10 do
  -- get rep of random img from the model
  local output = model:forward(img)

  -- calculate the MSE between natural img and random img ((maybe need normalise later))
  local loss = criterion:forward(output, target) 

  -- compute the derivative of the loss wrt the outputs of the model   dloss/dout
  local gradLoss = criterion:backward(output, target) 
  --print(torch.mean(gradLoss))
  --gradLoss:div(torch.pow(output, 2):mean())
  --print(torch.mean(gradLoss))
  local imgLoss = model:backward(img, gradLoss)

  img:add(imgLoss)
  image.display(img)
end


function feval = function ()




    -- return loss, grad
    local feval = function(x) -- x is parameters
      if x ~= params then
        params:copy(x)
      end
      grads:zero()

      -- forward
      local outputs = model:forward(data.inputs)
      local loss = criterion:forward(outputs, data.targets)
      -- backward
      local dloss_doutput = criterion:backward(outputs, data.targets)
      model:backward(data.inputs, dloss_doutput)

      return loss, grads
    end