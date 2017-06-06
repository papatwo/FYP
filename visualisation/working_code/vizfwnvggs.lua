local image = require 'image'
local loadcaffe = require 'loadcaffe'
require 'dpnn'
require 'TVCriterion'

-- Options
local imgSize = 224 -- Can be smaller than 224 x 224 input as only using convolutional layers

-- Load VGG (small)
local prototxt = '/home/akaishuushan/FYP/VGG_CNN_S/VGG_CNN_S_deploy.prototxt' -- define prototxt path
local caffemodel='/home/akaishuushan/FYP/VGG_CNN_S/VGG_CNN_S.caffemodel' -- define caffemodel path
vgg = loadcaffe.load(prototxt, caffemodel, 'nn') 
-- Set up image save path
svpath = '/home/akaishuushan/FYP/visualisation/results/filterviz'
-- pick up the layer idx of conv_layers
conv_layer = torch.Tensor({{1, 5, 8, 10, 12}})


-- Remove unrequired layers
for l = 3, conv_layer:size()[2] do -- 5
  model = nn.Sequential()
    for i = 1, conv_layer[1][l] do -- get the new model
      model:add(vgg:get(i))
    end
    print(model)
    img = torch.Tensor(3, imgSize, imgSize):uniform(-1, 1):mul(20):add(128)
    -- Display
    local win = image.display({image = img})
    filter_N = model:forward(img):size()[1]

  for filterIndex = 1, filter_N do
    -- Create white noise image
     img = torch.Tensor(3, imgSize, imgSize):uniform(-1, 1):mul(20):add(128)
    -- Create target gradient to maximise mean of filter
     gradLoss = torch.zeros(model:forward(img):size()) -- Work out size automatically from input and model
    
    
    --for filterIndex = 1, filter_N do
      --local filterIndex = f --80 -- Must index a valid convolutional filter
      gradLoss[filterIndex] = 1 -- Visualise one filter (all other  filter indices are 0 but the selected filter is 1)
      -- Maximise mean activation of filter
      for it = 1, 25 do -- Run gradient ascent for iterations
        local output = model:forward(img)
        local loss = torch.mean(output[filterIndex]) -- Mean activation of filter (which should be maximised)
        local imgLoss = model:backward(img, gradLoss) -- Gradient of input wrt (maximise activation) loss
        imgLoss:div(math.sqrt(torch.pow(imgLoss, 2):mean()) + 1e-5) -- Normalise gradient
        img:add(imgLoss) -- No range normalisation within loop
         norm_img = img:clone():div(255)
        image.display({image = norm_img, win = win})
      end
      image.save(paths.concat(svpath, l..filterIndex .. '.jpg'), norm_img) -- l: l th conv layer, f: f th filter
  end
   
end

  

  
  

  
