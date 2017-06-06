require 'nn'
require 'loadcaffe'
require 'optim'
require 'image'

--Four preprocessing steps:
--1. resize img
--2. crop img at 224x224
--3. change colour channel from RBG --> BGR
--4. rescale to 0-255 and subtract mean


-- 1. resize img

local loadSize = {3,256,256} --{colour channel, width, height}
local sampleSize = {3,224,224}

-- Load image
local path = '/home/akaishuushan/FYP/VGG_CNN_S/test_img/zbuk.jpg'
--local path = '/home/akaishuushan/下载/n02113186/n02113186_128.JPEG' --corgi
--local path = '/home/akaishuushan/下载/f4bdead1fda47f1 (1).jpg' --cat
--local path = '/home/akaishuushan/下载/f4bdead1fda47f1.jpg' --cat
--local path = '/home/akaishuushan/下载/images.jpg' -- truck
--local path = '/home/akaishuushan/下载/golden.jpg' -- golder retriver
--img=image.load(path,3,'float') -- different from image.load(path)<--double tensor! (maybe unecessary to define float, use image.load directly)
img=image.load(path,3)

-- resize the smallest side to 256 
-- !!!size() gives colour channel x height x width ranther than width x height
if img:size()[2]--[[height]]<img:size()[3]--[[width]] then --[[resize height to 256]]
-- image.scale(src, width, height)
img = image.scale(img, img:size()[3]*loadSize[2]/img:size()[2], loadSize[3])
else
img = image.scale(img, loadSize[2],img:size()[2]*loadSize[3]/img:size()[3] )
end


--2. crop img at 224x224

local ow=sampleSize[2] -- output image width
local oh=sampleSize[3] -- output image height
local iw=img:size()[3] -- output image width
local ih=img:size()[2] -- output image height
local w1=math.ceil((iw-ow)/2) -- width difference between origin and crop
local h1=math.ceil((ih-oh)/2) -- height difference between origin and crop

crop_img=image.crop(img,w1,h1,ow+w1,oh+h1) -- centre crop



-- 3. colour channel RGB --> BGR

-- input colour channels: 1-->R	     2-->G	3-->B
img_BGR=crop_img:clone() --copy the input original img
img_BGR[{1,{},{}}]=crop_img[{3,{},{}}] -- change B to the 1st channel
img_BGR[{3,{},{}}]=crop_img[{1,{},{}}] -- change R to the 3rd channel
				  -- keep G as before


-- 4. rescale to 0-255 and subtract mean

img_BGR:mul(255); -- torch express pixels 0-1, VGG requires 0-255

local bgr_means = {103.939,116.779,123.68} 
for i=1,3 do
img_BGR[{i,{},{}}]:add(-bgr_means[i])
end


----------------------------------------
-- Load model:
local prototxt='/home/akaishuushan/FYP/VGG_CNN_S/VGG_CNN_S_deploy.prtotxt'
local caffemodel='/home/akaishuushan/FYP/VGG_CNN_S/VGG_CNN_S.caffemodel'

model=loadcaffe.load(prototxt,caffemodel,'nn')

-- test 
predict=model:forward(img_BGR) -- obtain predicted probability of each class

x,inx=torch.max(predict,1) -- find the max probability and its index

-- map to labels
labeldir='/home/akaishuushan/FYP/VGG_CNN_S'
labels=paths.dir(labeldir)

label_table={}
for line in io.lines 'label.txt' do -- read in label txt
table.insert(label_table, line)
end

val,classes=predict:view(-1):sort(true) -- create a view of prediction results and sort it by descending. 【sort()是ascending】

--r=classes[1]+1 -- the first term of classes is the index of max probability, however label table index starts from 0, to map to the table this index should be added 1.
print('This image is',label_table[classes[1]],label_table[classes[2]],label_table[classes[3]] )
print(val[1], val[2], val[3])


	





