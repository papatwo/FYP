local image = require 'image'
local loadcaffe = require 'loadcaffe'
require 'dpnn'
require 'TVCriterion'
require 'reducenet'
--require 'preprocess'
require 'preproillu'
require 'alphanorm'
require 'cutorch'
require 'cunn'
require 'hdf5'
require 'deproillu'

use_cuda = 1

-- Load VGG (small)
--local prototxt = '//data/users/bw1613/VGG_CNN_S_deploy.prototxt' -- define prototxt path
--local caffemodel='//data/users/bw1613/VGG_CNN_S.caffemodel' -- define caffemodel path

local prototxt = '/data/users/hz4213/illust2vec.prototxt' -- define prototxt path
local caffemodel='/data/users/hz4213/illust2vec_ver200.caffemodel' -- define caffemodel path
svpath = '/data/users/hz4213/results/invert_illu2vec'
vgg = loadcaffe.load(prototxt, caffemodel, 'nn') 
img = image.load('fox.jpg')
meanFile = hdf5.open('/data/users/hz4213/FYP/visualisation/working_code/img_mean.h5','r')
img_mean = meanFile:read('img_mean'):all() 
meanFile:close()



-- Set up image save path
svpath = '/data/users/hz4213/results/filterviz_vgg16'
-- pick up the layer idx of conv_layers
conv_layer = torch.Tensor({{--[[1, 4, 7, 9,]] 12--[[,14, 17, 19, 22, 24, ]]}})

-- Remove unrequired layers
for l = 1, conv_layer:size()[2] do -- 5
  	model = nn.Sequential()
    	for i = 1, conv_layer[1][l] do -- get the new model
      		model:add(vgg:get(i))
    	end
    	print(model)
      model:cuda()
    	img = image.load('misaka.jpg')
    	img = preproillu(img,img_mean)
    	img = img:cuda()
    	-- Display
    	local win = image.display({image = img})
    
     		--gradLoss = torch.zeros(model:forward(img):size()) -- Work out size automatically from input and model
        	--gradLoss = gradLoss:cuda()
    
    		--for filterIndex = 1, filter_N do
      		--local filterIndex = f --80 -- Must index a valid convolutional filter
      		--gradLoss[filterIndex] = 1 -- Visualise one filter (all other  filter indices are 0 but the selected filter is 1)
      		-- Maximise mean activation of filter
      		for it = 1, 100 do -- Run gradient ascent for iterations
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
      		back = deproillu(img,img_mean:cuda())
      		image.display(back)
      		--image.save(paths.concat(svpath, l..'_'..filterIndex .. '.jpg'), norm_img) -- l: l th conv layer, f: f th filter

end

  

  
  

  
