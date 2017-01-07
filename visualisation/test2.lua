local image = require 'image'
local loadcaffe = require 'loadcaffe'
local optim = require 'optim'
require 'dpnn'
require 'nn' 
require 'torch'


local prototxt = '/home/akaishuushan/FYP/VGG_CNN_S/VGG_CNN_S_deploy.prototxt' -- define prototxt path
local caffemodel='/home/akaishuushan/FYP/VGG_CNN_S/VGG_CNN_S.caffemodel' -- define caffemodel path
vgg = loadcaffe.load(prototxt, caffemodel, 'nn') 

local loadSize = {3,256,256} --{colour channel, width, height}
local sampleSize = {3,224,224}
local img = image.load('golden.jpg')
losses = {}

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


-- 4. rescale to 0-255 and subtract mean

img_BGR:mul(255); -- torch express pixels 0-1, VGG requires 0-255

local bgr_means = {103.939,116.779,123.68} 
for i=1,3 do
img_BGR[{i,{},{}}]:add(-bgr_means[i])
end
image.display(img)
image.display(img_BGR)




local target = vgg:forward(img_BGR) :clone() -- to store the target image  out from model


randimg = torch.Tensor(3,224,224):normal(0,0.5)
--image.display(randimg)


criterion = nn.MSECriterion()

for i = 1, 2 do
	local output = vgg:forward(randimg)
	local loss = criterion:forward(output, target)

	local gradloss = criterion:backward(output, target)
	--local reg = torch.ones(#gradloss):mul(1e-5)
	gradloss:div(torch.pow(gradloss, 2):mean():sqrt():add(1e-5))

	local imgloss = vgg:backward(randimg, gradloss)
	randimg:add(imgloss)
	--losses [#losses + 1] =   loss
	--print(torch.mean(gradloss))

	-- normalise tensor: centre on 0 and ensure std is 0.1
	randimg:sub(torch.mean(randimg))
	randimg:mean()
	randimg:div(randimg:std():add(1e-5))
	randimg:mul(0.1)
	image.display(randimg)
end

    -- return loss, grad
     --local feval = function(randimg) -- x is parameters
 --[[ local  feval =  function (rand_in)
    --[[ if x ~= params then
        params:copy(x)
      end
      grads:zero()

      -- forward
      local output = vgg:forward(rand_in)
      local loss = criterion:forward(output, target)
      -- backward
      local gradLoss = criterion:backward(output, target)
      local imgLoss = vgg:backward(output, gradLoss)

      randimg:add(imgLoss)
      return loss, rand_in
    end]]

sgd_params = {	--train model using SGD
   weightDecay = 0, learningRate = 1e-3,   --10^(-3); decay regularizes solution / L2
   momentum = 0, learningRateDecay = 1e-4, --10^(-4); momentum averages steps over time
}

--loss, noise_img = optim.sgd(feval, randimg, sgd_params)

--[[for i =  1,10 do
	loss, noise_img = optim.sgd(feval, randimg, sgd_params)
	--loss, noise = feval(randimg)
	losses [#losses + 1] =   loss 
	--image.display(randimg)
end]]