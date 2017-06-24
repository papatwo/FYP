require 'nn'
require 'image'
local image = require 'image'
local loadcaffe = require 'loadcaffe'
require 'dpnn'
require 'TVCriterion'

-- Load VGG (small)
local prototxt = '/home/akaishuushan/FYP/VGG_CNN_S/VGG_CNN_S_deploy.prototxt' -- define prototxt path
local caffemodel='/home/akaishuushan/FYP/VGG_CNN_S/VGG_CNN_S.caffemodel' -- define caffemodel path
local vgg = loadcaffe.load(prototxt, caffemodel, 'nn') 

-------------------------------------------------------
-- Remove fully connected layers
function  reducenet( net, layer )
    for l = 1, layer do
        vgg:remove()
    end
    return vgg
end

criterion = nn.MSECriterion()
------------------------------------------------------

--local Normalization = {mean = 0/255, std = 0/255}
--[[function pre_norm(img)
    img_new = img
    img_new = img_new:div(255.0)
    img_new = img_new:add(-Normalization.mean)
    img_new = img_new:div(Normalization.std)
    return img_new
end

function post_norm(img)
    img_new = img * Normalization.std
    img_new = img_new:add(Normalization.mean)
    img_new = img_new:mul(255.0)
    img_new = image.scale(img_new, 500, 500, 'bicubic')
    return img_new
end]]

function preprocess( img )
    local loadSize = {3,256,256} --{colour channel, width, height}
    local sampleSize = {3,224,224}

    local loadSize = {3,256,256} 
    if img:size()[2]--[[height]] < img:size()[3]--[[width]] then --[[resize height to 256]]
    -- image.scale(src, width, height)
    img = image.scale(img, img:size()[3]*loadSize[2]/img:size()[2], loadSize[3])
    else
    img = image.scale(img, loadSize[2], img:size()[2]*loadSize[3]/img:size()[3])
    end

    local ow=sampleSize[2] -- output image width
    local oh=sampleSize[3] -- output image height
    local iw=img:size()[3] -- output image width
    local ih=img:size()[2] -- output image height
    local w1=math.ceil((iw-ow)/2) -- width difference between origin and crop
    local h1=math.ceil((ih-oh)/2) -- height difference between origin and crop

    local crop_img=image.crop(img,w1,h1,ow+w1,oh+h1) -- centre crop

    -- 3. colour channel RGB --> BGR 
    -- input colour channels: 1-->R      2-->G  3-->B
    local img_BGR=crop_img:clone() --copy the input original img
    img_BGR[{1,{},{}}]=crop_img[{3,{},{}}] -- change B to the 1st channel
    img_BGR[{3,{},{}}]=crop_img[{1,{},{}}] -- change R to the 3rd channel
                      -- keep G as before

    -- 4. rescale to 0-255 and subtract mean
    img_BGR:mul(img_BGR, 255) -- torch express pixels 0-1, VGG requires 0-255
    local bgr_means = {103.939,116.779,123.68}
    --local bgr_means = {0.4076,0.45796,0.48502} 
    for i=1,3 do
    img_BGR[i]:add(-bgr_means[i])
    end
    return img_BGR
end
----------------------------------------------------------------------------------------

img = image.load('golden.jpg')
img_copy = img:clone()
local Normalization = {mean = 118.380948/255, std = 61.896913/255}
--Normalization.mean = torch.mean(img)
--Normalization.std = torch.std(img)
img = preprocess(img)
function make_step(net, img, clip, step_size, jitter)
    local step_size = step_size or 0.01
    local jitter = jitter or 32
    local clip = clip
    if clip == nil then clip = true end

    local ox = 2*jitter - math.random(jitter)
    local oy = 2*jitter - math.random(jitter)
    img = image.translate(img, ox, oy) -- apply jitter shift

    local dst, g
    dst = net:forward(img)
    --Set the output gradients at the outermost layer to be equal to the outputs (So they keep getting amplified)
    g = net:updateGradInput(img, dst) --Computing the gradient of the module with respect to its own input
    --whether optimization (backpropagation) needed????? == objective(dis)

    -- Gradient ascent: apply normalized ascent step to the input image
    img:add(g:mul(step_size/torch.abs(g):mean())) -- get closer to our target data

    img = image.translate(img,-ox,-oy) -- unshift image jitter

    if clip then
        bias = Normalization.mean/Normalization.std
        --local bias = bgr_means
        img:clamp(-bias,1/Normalization.std-bias) --??or 255-bias
        --img:clamp(-bias, 255-bias)
    end
    return img
end

--Next implement an ascent through different scales. We call these scales "octaves".
function deepdream(net, base_img, iter_n, octave_n, octave_scale, end_layer, clip, visualize)
    -- set-up params
    local iter_n = iter_n or 10
    local octave_n = octave_n or 4
    local octave_scale = octave_scale or 1.4
    local end_layer = end_layer or 11
    local net = net
    local clip = clip
    if clip == nil then clip = true end
    cur_scale = octave_scale


    -- prepare base images for all octaves
    --local octaves = preprocess(base_img) 


    local octaves = {}     --So, the octaves is an array of images
    --initialized with the original image transformed into caffe's format
    octaves[octave_n] = img  -- *try comment out
    local _, h, w = unpack(base_img:size():totable())

    for i = octave_n-1,1,-1 do
        -- rescale (down???) from the last octave and store the new one in the former table content
        octaves[i] = image.scale(octaves[i+1], math.ceil((cur_scale)*w), math.ceil((cur_scale)*h),'simple')
        cur_scale = cur_scale * octave_scale
    end --this creates smaller versions of the images
           --One image for each octave.


    local src --to copy original img again
    local detail --allocate image for previous network-produced details

    for octave, octave_base in pairs(octaves) do --Iterate over the list of images                                                                   
        src = octave_base -- take the current octave image
        local _, h1, w1 = unpack(src:size():totable())  --Take the width and height of the current image
        if octave > 1 then --if it's not the smallest (first) octave
            -- upscale details from the previous octave
            detail = image.scale(detail, math.ceil(w1), math.ceil(h1),'simple')
            -- is it necessary to reshape src back to input image size
            src:add(detail)
        end

        for i=1,iter_n do --number of step iterations
            src = make_step(net, src, clip)

            -- visualization
            if visualize then
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

--[[
-- Load VGG (small)
local prototxt = '/home/akaishuushan/FYP/VGG_CNN_S/VGG_CNN_S_deploy.prototxt' -- define prototxt path
local caffemodel='/home/akaishuushan/FYP/VGG_CNN_S/VGG_CNN_S.caffemodel' -- define caffemodel path
local vgg = loadcaffe.load(prototxt, caffemodel, 'nn') 


local Normalization = {mean = 118.380948/255, std = 61.896913/255}
for l = 1, 11 do
  vgg:remove()
end
local img = image.load('golden.jpg')

function make_step(net, img, clip,step_size, jitter)
    local step_size = step_size or 0.01
    local jitter = jitter or 32
    local clip = clip
    if clip == nil then clip = true end

    local ox = 0--2*jitter - math.random(jitter)
    local oy = 0--2*jitter - math.random(jitter)
    img = image.translate(img,ox,oy) -- apply jitter shift
    local dst, g
    dst = net:forward(img)
    g = net:updateGradInput(img,dst)
    -- apply normalized ascent step to the input image
    img:add(g:mul(step_size/torch.abs(g):mean()))


    img = image.translate(img,-ox,-oy) -- apply jitter shift
    if clip then
        bias = Normalization.mean/Normalization.std
        img:clamp(-bias,1/Normalization.std-bias)
    end
    return img
end

function deepdream(net, base_img, iter_n, octave_n, octave_scale, end_layer, clip, visualize)

    local iter_n = iter_n or 10
    local octave_n = octave_n or 4
    local octave_scale = octave_scale or 1.4
    local end_layer = end_layer or 20
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

end]]


x = make_step(vgg,img)
image.display(x)