require 'nn'
local model = {}
-- warning: module 'data [type 5]' not found
table.insert(model, {'conv1', nn.SpatialConvolution(3, 96, 11, 11, 4, 4, 0, 0)})
table.insert(model, {'relu1', nn.ReLU(true)})
table.insert(model, {'norm1', nn.SpatialCrossMapLRN(5, 0.000100, 0.7500, 1.000000)})
table.insert(model, {'pool1', nn.SpatialMaxPooling(3, 3, 2, 2, 0, 0):ceil()})
-- warning: module 'conv2 [type 4]' not found
table.insert(model, {'relu2', nn.ReLU(true)})
table.insert(model, {'norm2', nn.SpatialCrossMapLRN(5, 0.000100, 0.7500, 1.000000)})
table.insert(model, {'pool2', nn.SpatialMaxPooling(3, 3, 2, 2, 0, 0):ceil()})
table.insert(model, {'conv3', nn.SpatialConvolution(256, 384, 3, 3, 1, 1, 1, 1)})
table.insert(model, {'relu3', nn.ReLU(true)})
-- warning: module 'conv4 [type 4]' not found
table.insert(model, {'relu4', nn.ReLU(true)})
-- warning: module 'conv5 [type 4]' not found
table.insert(model, {'relu5', nn.ReLU(true)})
table.insert(model, {'pool5', nn.SpatialMaxPooling(3, 3, 2, 2, 0, 0):ceil()})
table.insert(model, {'torch_view', nn.View(-1):setNumInputDims(3)})
table.insert(model, {'fc6', nn.Linear(9216, 4096)})
table.insert(model, {'relu6', nn.ReLU(true)})
table.insert(model, {'drop6', nn.Dropout(0.500000)})
table.insert(model, {'fc7', nn.Linear(4096, 4096)})
table.insert(model, {'relu7', nn.ReLU(true)})
table.insert(model, {'drop7', nn.Dropout(0.500000)})
table.insert(model, {'fc8', nn.Linear(4096, 1000)})
table.insert(model, {'loss', nn.SoftMax()})
return model