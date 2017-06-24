local image = require 'image'
local loadcaffe = require 'loadcaffe'
require 'dpnn'
require 'TVCriterion'
require 'reducenet'
require 'preprocess'
require 'alphanorm'
require 'cutorch'

use_cuda = 1

-- Load VGG (small)
--local prototxt = '//data/users/bw1613/VGG_CNN_S_deploy.prototxt' -- define prototxt path
--local caffemodel='//data/users/bw1613/VGG_CNN_S.caffemodel' -- define caffemodel path

local prototxt = '/data/users/bw1613/FYP/VGG_CNN_S/VGG_CNN_S_deploy.prototxt' -- define prototxt path
local caffemodel='/data/users/bw1613/FYP/VGG_CNN_S/VGG_CNN_S.caffemodel' -- define caffemodel path

vgg = loadcaffe.load(prototxt, caffemodel, 'nn') 
img = image.load('golden.jpg')
if use_cuda then
	img = img:cuda()
end
img = preprocess(img)
local win = image.display({image = img})

imgW = 224--img:size()[3]
imgH = 224--img:size()[2]

mse = nn.MSECriterion()
alpha_idx = 6 -- the alpha value of alpha norm
TVCriterion = nn.TVCriterion()
alphanorm = nn.alphanorm(alpha_idx)
net = nn.Sequential()
net:add(alphanorm())
net:add(TVCriterion())

for l = 1, 1 do
        	net:add(vgg:get(l))
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
for n = 1, 1 do -- do iterations to visualise
	--net:zeroGradParameters() -- try have/without this line: if the module has params, this will zero the accumulation of the gradients wrt these params
	net:evaluate()
	x_f = net:forward(x_n) --inversion feature from this layer
	loss = mse:forward(x_f, x_0f) -- compute the MSE loss of the inversion img
	grad_loss = mse:backward(x_f, x_0f) -- compute the grad of loss wrt to x_f
	inv_grad = net:backward(x, grad_loss) -- Gradient of input wrt (maximise activation) activation 
	print(torch.sum(inv_grad))
	--inv_grad:div(math.sqrt(torch.pow(inv_grad, 2):mean()) + 1e-5)
	-- grad descent bit
	x_n:add(-inv_grad)
	--x:add(-(inv_grad/x0_norm + 0.1 * alpha_grad + 0.1 * beta_grad)) --use grad wrt to these to optimise input
	image.display({image = x_n, win = win})
end


	



