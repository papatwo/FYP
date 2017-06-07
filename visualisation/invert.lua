local image = require 'image'
local loadcaffe = require 'loadcaffe'
require 'dpnn'
require 'nn'
require 'TVCriterion'
require 'reducenet'
require 'preprocess'
require 'alphanorm'


-- Load VGG (small)
local prototxt = '/home/akaishuushan/FYP/VGG_CNN_S/VGG_CNN_S_deploy.prototxt' -- define prototxt path
local caffemodel='/home/akaishuushan/FYP/VGG_CNN_S/VGG_CNN_S.caffemodel' -- define caffemodel path
vgg = loadcaffe.load(prototxt, caffemodel, 'nn') 
img = image.load('golden.jpg')
img = preprocess(img)
local win = image.display({image = img})

imgW = 224--img:size()[3]
imgH = 224--img:size()[2]

mse = nn.MSECriterion()
alpha_idx = 6 -- the alpha value of alpha norm
alpha_weight = 0.00005
TVCriterion = nn.TVCriterion(0.00005)
alphanorm = nn.alphanorm(alpha_idx, alpha_weight)
net = nn.Sequential()
net:add(TVCriterion)
net:add(alphanorm)
for l = 1, 12 do
        	net:add(vgg:get(l))
end

    	-- compute the features of reference image x_0
-- forward pass: through all conv layers
net:evaluate()
f = net:forward(img) -- reference image feature computed from this layer
x_0f = f:clone()
x0_norm = math.sqrt(torch.sum(torch.pow(x_0f, 2))) --scalar_l2_norm_of_x_0f
x = torch.Tensor(3, imgH, imgW):uniform(-1, 1):mul(20):add(128) --initial inversion img
x_n = torch.div(x, 255)
for n = 1, 4000 do -- do iterations to visualise
	--net:zeroGradParameters() -- try have/without this line: if the module has params, this will zero the accumulation of the gradients wrt these params
	net:evaluate()
	x_f = net:forward(x_n) --inversion feature from this layer
	loss = mse:forward(x_f, x_0f) -- compute the MSE loss of the inversion img
	grad_loss = mse:backward(x_f, x_0f) -- compute the grad of loss wrt to x_f
	--grad_loss:div(x0_norm)
	inv_grad = net:backward(x_n, grad_loss) -- Gradient of input wrt (maximise activation) activation 
	print(torch.sum(inv_grad))
	--inv_grad:div(math.sqrt(torch.pow(inv_grad, 2):mean()) + 1e-5)
	-- grad descent bit
	x_n:add(-inv_grad)
	--alpha_grad = alphanorm:backward(x_n, inv_grad)
	--beta_grad = TVCriterion:backward(x_n,inv_grad)
	--x_n:add(-(inv_grad/x0_norm + 0.1 * alpha_grad + 0.1 * beta_grad)) --use grad wrt to these to optimise input
	image.display({image = x_n, win = win})
end