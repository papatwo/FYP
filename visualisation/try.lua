require "nn"
require "image"
require "math"
require "loadcaffe"
require 'dpnn'


-- Set this from the input image first before calling pre_process
local Normalization = {mean = 0/255, std = 0/255}


function reducenet(net, layer)
	local network = nn.Sequential()
	for i=1,layer do
		network:add(net:get(i))
	end
	return network
end

-- Make sure Normalization is set
function pre_process(img)
	img_new = img:float()
	img_new = img_new:div(255.0)
	img_new = img_new:add(-Normalization.mean)
	img_new = img_new:div(Normalization.std)
	return img_new
end

function post_process(img)
	img_new = img * Normalization.std
	img_new = img_new:add(Normalization.mean)
	img_new = img_new:mul(255.0)
	img_new = image.scale(img_new, 500, 500, 'bicubic')
	return img_new
end



local prototxt = '/home/akaishuushan/FYP/VGG_CNN_S/VGG_CNN_S_deploy.prototxt' -- define prototxt path
local caffemodel='/home/akaishuushan/FYP/VGG_CNN_S/VGG_CNN_S.caffemodel' -- define caffemodel path
local vgg = loadcaffe.load(prototxt, caffemodel, 'nn') 

netw = reducenet(vgg,18)
print(netw)


input = image.load('golden.jpg',3,'byte')
input_copy = image.load('golden.jpg',3,'byte')
Normalization.mean = torch.mean(input:float())/255.0
Normalization.std = torch.std(input:float())/255.0
print(Normalization.mean .. " " .. Normalization.std)

input = pre_process(input)
input_copy = pre_process(input_copy)

-- Generally networks use 3x244x244 images as inputs
input = image.scale(input,224,224)
input_copy = image.scale(input_copy,224,224)
-- scale down h and w by the same factor and crop the central region to 224 by 224

local win = image.display{image=(post_process(input))}

local total_octaves = 2
local drop_scale = 1.25/1.4
cur_drop_scale = drop_scale

local octaves = {}
octaves[total_octaves] = input:float()
local base_size = input:size()
for j=total_octaves-1,1,-1 do
    local cur_octave = image.scale(octaves[j+1], math.floor(cur_drop_scale*base_size[2]), math.floor(cur_drop_scale*base_size[3]),'bicubic')
    cur_drop_scale = cur_drop_scale * drop_scale
    octaves[j] = cur_octave
end

local prev_change
local final_img

for oct=1,total_octaves do
	local cur_oct = octaves[oct]
	local cur_size = cur_oct:size()
	prev_change = input:clone()

	if oct > 1 then
		prev_change = image.scale(prev_change, cur_size[2], cur_size[3], 'bicubic')
		cur_oct:add(prev_change)
	end
	cur_oct = image.scale(cur_oct, 224, 224, 'bicubic')
	print(cur_oct:size())
	
	for tt=1,10 do
	    -- Forward prop in the neural network
	    local outputs_cur = netw:forward(cur_oct:double())
	    -- Set the output gradients at the outermost layer to be equal to the outputs (So they keep getting amplified)
	    local output_grads = outputs_cur
	    local inp_grad = netw:updateGradInput(cur_oct:double(),output_grads)
	    print(cur_oct:type())
	    -- Gradient ascent
	    cur_oct = cur_oct:double():add(inp_grad:mul(0.1/torch.abs(inp_grad):mean()))
	    image.display{image=(post_process(cur_oct)), win = win}
	    print(oct,tt)
	end
	cur_oct = cur_oct:float()
	cur_oct = image.scale(cur_oct, cur_size[2], cur_size[3], 'bicubic')
	prev_change = cur_oct - octaves[oct]
	final_img = cur_oct
	print(cur_oct:size())
end

--image.display{image=(post_process(final_img))}
print('Done processing')