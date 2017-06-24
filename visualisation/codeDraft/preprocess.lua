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