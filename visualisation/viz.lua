local image = require 'image'
local loadcaffe = require 'loadcaffe'
require 'dpnn'
require 'TVCriterion'

local model = nn.Sequential()
local vgg = loadcaffe.load('VGG_CNN_S_deploy.prototxt', 'VGG_CNN_S.caffemodel', 'nn')
model:add(nn.TVCriterion(0))
model:add(nn.MulConstant(255, true))
model:add(nn.SplitTable(1, 3))

local shuffleModel = nn.ConcatTable()

local BModel = nn.Sequential()
BModel:add(nn.SelectTable(3)) -- Select B channel
BModel:add(nn.View(-1, 1, 224, 224)) -- Add back channel dimension
BModel:add(nn.AddConstant(103.939, true)) -- Subtract B mean pixel value

local GModel = nn.Sequential()
GModel:add(nn.SelectTable(2)) -- Select G channel
GModel:add(nn.View(-1, 1, 224, 224)) -- Add back channel dimension
GModel:add(nn.AddConstant(116.779, true)) -- Subtract G mean pixel value

local RModel = nn.Sequential()
RModel:add(nn.SelectTable(2)) -- Select R channel
RModel:add(nn.View(-1, 1, 224, 224)) -- Add back channel dimension
RModel:add(nn.AddConstant(123.68, true)) -- Subtract R mean pixel value

shuffleModel:add(BModel)
shuffleModel:add(GModel)
shuffleModel:add(RModel)
model:add(shuffleModel) -- RGB -> BGR shuffle
model:add(nn.JoinTable(1, 3))
model:add(vgg)

local crit = nn.MSECriterion()

local img = torch.Tensor(3, 224, 224):normal(0, 0.5)
image.display(img)

local targetImage = image.load('gingerkitten.jpg')
local target = model:forward(targetImage)
--local target = torch.zeros(1000)
--target[1] = 1

for i = 1, 10 do
  local output = model:forward(img)
  local loss = crit:forward(output, target)
  local gradLoss = crit:backward(output, target)
  --print(torch.mean(gradLoss))
  --gradLoss:div(torch.pow(output, 2):mean())
  --print(torch.mean(gradLoss))
  local imgLoss = model:backward(img, gradLoss)

  img:add(imgLoss)
  image.display(img)
end