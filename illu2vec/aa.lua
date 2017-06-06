-- this one can make different feature visualised but not very clear.
-- when VGG remove 12 layers can deliever very nice circle deep dream around eyes area

require 'nn'
require 'image'
local image = require 'image'
local loadcaffe = require 'loadcaffe'
require 'dpnn'
require 'TVCriterion'
-- Load VGG (small)
local prototxt = '/home/akaishuushan/FYP/illu2vec/illust2vec.prototxt' -- define prototxt path
local caffemodel='/home/akaishuushan/FYP/illu2vec/illust2vec_ver200.caffemodel' -- define caffemodel path
local vgg = loadcaffe.load(prototxt, caffemodel, 'nn') 
print(vgg)
print(vgg:size())

local Normalization = {mean = 118.380948/255, std = 61.896913/255}

--[[function reduceNet( full_net, end_layer )
	local net = nn.Sequential()
	for l = 1, end_layer do
		net:add(full_net:get(l))
	end
	return net
end]]

for l = 1, 10 do
  vgg:remove()
end
--local img= torch.rand(3,224,224):uniform()
local img = image.load('golden.jpg')
--local img = image.load('images.jpg')
--local img = image.load('circle.png')

function make_step(net, img, clip, step_size, jitter)

	-- gradient ascent step
    local step_size = step_size or 0.2
    local jitter = jitter or 32
    local clip = clip
    if clip == nil then clip = true end

    local ox = 0--2*jitter - math.random(jitter)
    local oy = 0--2*jitter - math.random(jitter)
    --apply jitter

    img = image.translate(img,ox,oy) -- apply jitter shift
    local dst, g
    dst = net:forward(img) 
    g = net:updateGradInput(img,dst)
    -- apply normalized ascent step to the input image
    img:add(g:mul(step_size/torch.abs(g):mean()))
    --img:add(g)

    --apply unshift jitter

    


    img = image.translate(img,-ox,-oy) -- apply jitter shift
    if clip then  --try comment the clamp out!!!!!!!
        bias = Normalization.mean/Normalization.std
        img:clamp(-bias,1/Normalization.std-bias)
    end
    return img
end

function deepdream(net, base_img, iter_n, octave_n, octave_scale, end_layer, clip, visualize)

    local iter_n = iter_n or 10 --15
    local octave_n = octave_n or 4 --3
    local octave_scale = octave_scale or 1.4
    --local end_layer = end_layer or 11
    --local net = reduceNet(net, end_layer)
    local net = net
    local clip = clip
    if clip == nil then clip = true end
    -- prepare base images for all octaves
    local octaves = {}
    octaves[octave_n] = torch.add(base_img, -Normalization.mean):div(Normalization.std)
    local _,h,w = unpack(base_img:size():totable())

    for i=octave_n-1,1,-1 do
        octaves[i] = image.scale(octaves[i+1], math.ceil((1/octave_scale)*w), math.ceil((1/octave_scale)*h),'simple')
    end


    local detail
    local src

    for octave, octave_base in pairs(octaves) do
        src = octave_base
        local _,h1,w1 = unpack(src:size():totable())
        if octave > 1 then
            -- upscale details from the previous octave
            detail = image.scale(detail, w1, h1,'simple')
            src:add(detail)
        end
        for i=1,iter_n do
            src = make_step(net, src, clip)
            if visualize then
                -- visualization
                vis = torch.mul(src, Normalization.std):add(Normalization.mean)

                if not clip then -- adjust image contrast if clipping is disabled
                    vis = vis:mul(1/vis:max())
                end

                image.display(vis)
            end
        end
        -- extract details produced on the current octave
        detail = src-octave_base
    end
    -- returning the resulting image
    src:mul(Normalization.std):add(Normalization.mean)
    return src

end


x = deepdream(vgg,img)
image.display(x)

---------------------------
--[[
--equivalent to np.roll
x=torch.zeros(2,4):uniform(0,10)
y = torch.zeros(2, 4):scatter(1, torch.LongTensor{{2,2,2,2},{1,1,1,1}}, x)
y = torch.zeros(2, 4):scatter(2, torch.LongTensor{{3, 4, 1, 2}, {3, 4, 1, 2}}, x)]]