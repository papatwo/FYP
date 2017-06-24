require 'nn'
--require 'cunn'
--require 'cudnn'
require 'image'
require "loadcaffe"


local cuda = false
local prototxt = '/home/akaishuushan/FYP/VGG_CNN_S/VGG_CNN_S_deploy.prototxt' -- define prototxt path
local caffemodel='/home/akaishuushan/FYP/VGG_CNN_S/VGG_CNN_S.caffemodel' -- define caffemodel path
--torch.setdefaulttensortype('torch.FloatTensor')
net = loadcaffe.load(prototxt, caffemodel, 'nn') 
--net = torch.load('./OverFeatModel.t7'):float()
--net:training()


local Normalization = {mean = 118.380948/255, std = 61.896913/255}

function reduceNet(full_net,end_layer)
    local net = nn.Sequential()
    for l=1,end_layer do
        net:add(full_net:get(l))
    end
    return net
end

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

function make_step(net, img, clip,step_size, jitter)
    local step_size = step_size or 0.01
    local jitter = jitter or 32
    local clip = clip
    if clip == nil then clip = true end

    local ox = 0--2*jitter - math.random(jitter)
    local oy = 0--2*jitter - math.random(jitter)
    img = image.translate(img,ox,oy) -- apply jitter shift
    local dst, g

    --img=preprocess(img)
    dst = net:forward(img)
    print("up to here!!!")
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
    local octave_n = octave_n or 3
    local octave_scale = octave_scale or 1.4
    local end_layer = end_layer or 20
    local net = reduceNet(net, end_layer)
    local clip = clip
    local visualize=true
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
        print("here",src:size())
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
        print("src",src:size())
        print("octbase", octave_base:size())
        -- extract details produced on the current octave
        detail = src-octave_base
    end
    -- returning the resulting image
    src:mul(Normalization.std):add(Normalization.mean)
    return src

end



img = image.load('golden.jpg')
img=preprocess(img)
x = deepdream(net,img)
--dst = net:forward(img)
--print(dst)
image.display(img)

