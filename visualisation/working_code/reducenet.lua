--[[require 'nn'
require 'image'
local image = require 'image'
local loadcaffe = require 'loadcaffe'
require 'dpnn'
require 'TVCriterion']]

-- Load VGG (small)
--[[local prototxt = '/home/akaishuushan/FYP/VGG_CNN_S/VGG_CNN_S_deploy.prototxt' -- define prototxt path
local caffemodel='/home/akaishuushan/FYP/VGG_CNN_S/VGG_CNN_S.caffemodel' -- define caffemodel path
local vgg = loadcaffe.load(prototxt, caffemodel, 'nn') ]]


function  reducenet( net, layer )
	print("full net", net)
	local temp = nn.Sequential()
	for l = 1, layer do
		temp:add(net:get(l))
	end
	print("reduced net",temp)
	return temp
end

--[[for l = 1, 8  do
  vgg:remove()
end
print(vgg)]]