local image = require 'image'
local loadcaffe = require 'loadcaffe'
require 'dpnn'
require 'TVCriterion'

-- Options
local imgSize = 224 -- Can be smaller than 224 x 224 input as only using convolutional layers
local filterIndex = 1 -- Must index a valid convolutional filter

-- Load VGG (small)
local prototxt = '/home/akaishuushan/FYP/VGG_CNN_S/VGG_CNN_S_deploy.prototxt' -- define prototxt path
local caffemodel='/home/akaishuushan/FYP/VGG_CNN_S/VGG_CNN_S.caffemodel' -- define caffemodel path
local vgg = loadcaffe.load(prototxt, caffemodel, 'nn') 
-- Remove fully connected layers
for l = 1, 11 do
  vgg:remove()
end
--print(vgg:get(vgg:size()))

local loadSize = {3,256,256} --{colour channel, width, height}
local sampleSize = {3,224,224}
local img = image.load('golden.jpg')

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
 --image.display(rescale(img))
 img = rescale(img)
 print(img:size())


--2. crop img at 224x224

local ow=sampleSize[2] -- output image width
local oh=sampleSize[3] -- output image height
local iw=img:size()[3] -- output image width
local ih=img:size()[2] -- output image height
local w1=math.ceil((iw-ow)/2) -- width difference between origin and crop
local h1=math.ceil((ih-oh)/2) -- height difference between origin and crop

local crop_img=image.crop(img,w1,h1,ow+w1,oh+h1) -- centre crop



-- 3. colour channel RGB --> BGR 

-- input colour channels: 1-->R	     2-->G	3-->B
local img_BGR=crop_img:clone() --copy the input original img
img_BGR[{1,{},{}}]=crop_img[{3,{},{}}] -- change B to the 1st channel
img_BGR[{3,{},{}}]=crop_img[{1,{},{}}] -- change R to the 3rd channel
				  -- keep G as before
				  
--img_BGR:mul(img_BGR, 255)				

-- 4. rescale to 0-255 and subtract mean

--img_BGR:mul(255) -- torch express pixels 0-1, VGG requires 0-255

--local bgr_means = {103.939,116.779,123.68}
local bgr_means = {0.4076,0.45796,0.48502} 
for i=1,3 do
img_BGR[i]:add(-bgr_means[i])
end
image.display(img)
image.display(img_BGR)

-------------------------------------------------------------------------

-- Create target gradient to maximise mean of filter
local gradLoss = torch.zeros(vgg:forward(img_BGR):size()) -- Work out size automatically from input and model
gradLoss[filterIndex] = 1 -- Visualise one filter
-- Display
local win = image.display({image = img_BGR})

-- Maximise mean activation of filter
for i = 1, 1 do
  local output = vgg:forward(img_BGR)
  local loss = torch.mean(output[filterIndex]) -- Mean activation of filter (which should be maximised)
  print(loss)
  local imgLoss = vgg:backward(img_BGR, gradLoss)
  imgLoss:div(math.sqrt(torch.pow(imgLoss, 2):mean()) + 1e-5) -- Normalise gradient

  img_BGR:add(imgLoss) -- No range normalisation within loop
  image.display({image = img_BGR, win = win})
end