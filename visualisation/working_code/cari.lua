local image = require 'image'
local loadcaffe = require 'loadcaffe'
require 'dpnn'
require 'TVCriterion'
require 'cutorch'
require 'cunn'
require 'preprocess'
require 'deprocess'

-- Options
local imgSize = 224 -- Can be smaller than 224 x 224 input as only using convolutional layers

-- Load VGG (small)
local prototxt = '/data/users/hz4213/VGG_16_deploy.prototxt' -- define prototxt path
local caffemodel='/data/users/hz4213/VGG_ILSVRC_16_layers.caffemodel' -- define caffemodel path

vgg = loadcaffe.load(prototxt, caffemodel, 'nn') 


-- Set up image save path
svpath = '/data/users/hz4213/results/filterviz_vgg16'
-- pick up the layer idx of conv_layers
conv_layer = torch.Tensor({{ --[[1, 3,]] 6--[[, 8, 11--[[, 13, 15, 18, 20 ,22, 25 27, 29]]}}) -- find number of conv _ly in the model and identify the layer index here

-- Remove unrequired layers
for l = 1, conv_layer:size()[2] do -- 5
  	model = nn.Sequential()
    	for i = 1, conv_layer[1][l] do -- get the new model
      		model:add(vgg:get(i))
    	end
    	print(model)
      model:cuda()
    	img = image.load('misaka.jpg')
    	img = preprocess(img)
    	img = img:cuda()
    	-- Display
    	local win = image.display({image = img})
    
     		--gradLoss = torch.zeros(model:forward(img):size()) -- Work out size automatically from input and model
        	--gradLoss = gradLoss:cuda()
    
    		--for filterIndex = 1, filter_N do
      		--local filterIndex = f --80 -- Must index a valid convolutional filter
      		--gradLoss[filterIndex] = 1 -- Visualise one filter (all other  filter indices are 0 but the selected filter is 1)
      		-- Maximise mean activation of filter
      		for it = 1, 200 do -- Run gradient ascent for iterations
      			print(it)
        			 output = model:forward(img)
               		 gradLoss = torch.lt(output,0)
               		 gradLoss = gradLoss:cuda()
               		 gradLoss = gradLoss:cdiv(output)
               --gradLoss=0 -- for caricaturization
        			 loss = torch.mean(output) -- Mean activation of filter (which should be maximised)
        			local imgLoss = model:backward(img, gradLoss) -- Gradient of input wrt (maximise activation) loss
        			imgLoss:div(math.sqrt(torch.pow(imgLoss, 2):mean()) + 1e-5) -- Normalise gradient
        			img:add(imgLoss) -- No range normalisation within loop
         			norm_img = img:clone():div(255)
        			image.display({image = norm_img, win = win})
      		end
      		back = deprocess(img)
      		image.display(back)
      		--image.save(paths.concat(svpath, l..'_'..filterIndex .. '.jpg'), norm_img) -- l: l th conv layer, f: f th filter

end

  

  
  

  
