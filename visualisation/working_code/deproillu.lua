function deproillu( img, img_mean )
    
    -- 3. colour channel RGB --> BGR 
    -- input colour channels: 1-->R      2-->G  3-->B
    local back=img:clone() --copy the input original img
    back[{1,{},{}}]=img[{3,{},{}}] -- change B to the 1st channel
    back[{3,{},{}}]=img[{1,{},{}}] -- change R to the 3rd channel
                      -- keep G as before
--add subtracted mean back[[
    -- 4. rescale to 0-255 and subtract mean
    back:mul(255)
    local rgb_means = img_mean
    --local bgr_means = {0.4076,0.45796,0.48502} 
    for i=1,3 do
        back[i]:add(rgb_means[i])
    end
    return back
end