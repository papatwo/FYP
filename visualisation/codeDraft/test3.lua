local image = require 'image'
local loadcaffe = require 'loadcaffe'
require 'dpnn'
require 'TVCriterion'

-- Options
local imgSize = 224 -- Can be smaller than 224 x 224 input as only using convolutional layers
local filterIndex = 256 --80 -- Must index a valid convolutional filter

-- Load VGG (small)
local prototxt = '/home/akaishuushan/FYP/VGG_CNN_S/VGG_CNN_S_deploy.prototxt' -- define prototxt path
local caffemodel='/home/akaishuushan/FYP/VGG_CNN_S/VGG_CNN_S.caffemodel' -- define caffemodel path
local vgg = loadcaffe.load(prototxt, caffemodel, 'nn') 

-- Remove fully connected layers
for l = 1, 13 do
  vgg:remove()
end
-- print(vgg:get(vgg:size()))


-- Construct preprocessing network
local model = nn.Sequential()
--model:add(nn.TVCriterion(0))
model:add(vgg)


-- Create white noise image
--local img = torch.Tensor(3, imgSize, imgSize):uniform(-1, 1):add(100)
local img = torch.Tensor(3, imgSize, imgSize):uniform(-1, 1):mul(20):add(128)
-- Create target gradient to maximise mean of filter
local gradLoss = torch.zeros(model:forward(img):size()) -- Work out size automatically from input and model
gradLoss[filterIndex] = 1 -- Visualise one filter (all other  filter indices are 0 but the selected filter is 1)
-- Display
local win = image.display({image = img})

-- Set up image save path
--ocal svpath = '/home/akaishuushan/FYP/visualisation/results/'
-- Maximise mean activation of filter
for i = 1, 50 do -- Run gradient ascent for iterations
  local output = model:forward(img)
  local loss = torch.mean(output[filterIndex]) -- Mean activation of filter (which should be maximised)
  local imgLoss = model:backward(img, gradLoss) -- Gradient of input wrt (maximise activation) loss
  imgLoss:div(math.sqrt(torch.pow(imgLoss, 2):mean()) + 1e-5) -- Normalise gradient
  img:add(imgLoss) -- No range normalisation within loop
  --print(img[1])
--  image.display({image = img, win = win})
  local norm_img = img:clone():div(255)
  image.display({image = norm_img, win = win})

  --image.save(paths.concat(svpath, i .. '.jpg'), norm_img) -- '..' might be the connector between file name and ext!
end