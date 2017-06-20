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

img = preproillu(img, img_mean)
if use_cuda then
	img = img:cuda()
end
local win = image.display({image = img})

imgW = 256--img:size()[3]
imgH = 256--img:size()[2]

mse = nn.MSECriterion()
mse:cuda()
alpha_idx = 6 -- the alpha value of alpha norm
alpha_weight = 0.00005
TVCriterion = nn.TVCriterion(0.00005)
alphanorm = nn.alphanorm(alpha_idx, alpha_weight)


conv_layer = torch.Tensor({{1, 4, 7, 9, 12, 14, 17, 19, 22, 24, 26}})


for l = 1, conv_layer:size()[2] do
	net = nn.Sequential()
    --net:add(alphanorm)
    --net:add(TVCriterion)
	for i = 1, conv_layer[1][l] do
    	net:add(vgg:get(i))
	end
    
    if use_cuda then
	net:cuda()
	end
    -- compute the features of reference image x_0
	-- forward pass: through all conv layers
	net:evaluate()
	f = net:forward(img) -- reference image feature computed from this layer
	x_0f = f:clone()
	x0_norm = math.sqrt(torch.sum(torch.pow(x_0f, 2))) --scalar_l2_norm_of_x_0f
	if use_cuda then
		x = torch.CudaTensor(3, imgH, imgW):uniform(-1, 1):mul(20):add(128) --initial inversion img
	else
		x = torch.Tensor(3, imgH, imgW):uniform(-1, 1):mul(20):add(128) --initial inversion img
	end
	x_n = torch.div(x, 255)
	for n = 1, 30000 do -- do iterations to visualise
	--net:zeroGradParameters() -- try have/without this line: if the module has params, this will zero the accumulation of the gradients wrt these params
		net:evaluate()
		x_f = net:forward(x_n) --inversion feature from this layer
		loss = mse:forward(x_f, x_0f) -- compute the MSE loss of the inversion img
		grad_loss = mse:backward(x_f, x_0f) -- compute the grad of loss wrt to x_f
		inv_grad = net:backward(x, grad_loss) -- Gradient of input wrt (maximise activation) activation 
		print(n)
		--inv_grad:div(math.sqrt(torch.pow(inv_grad, 2):mean()) + 1e-5)
		-- grad descent bit
		x_n:add(-inv_grad)
		--x:add(-(inv_grad/x0_norm + 0.1 * alpha_grad + 0.1 * beta_grad)) --use grad wrt to these to optimise input
		image.display({image = x_n, win = win})
	end
	img_mean = img_mean:cuda()
	back = deproillu(x_n,img_mean)
	image.display({image = back--[[, win = win]]})
    image.save(paths.concat(svpath, l .. '.jpg'), back)
end



	



