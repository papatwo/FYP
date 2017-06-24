-- Adapted from jcjohnson/neural-style

local TVCriterion, parent = torch.class('nn.TVCriterion', 'nn.Module')

function TVCriterion:__init(strength)
  parent.__init(self)
  self.enabled = true
  self.sizeAverage = true
  self.strength = strength or 1
  self.xDiff = torch.Tensor()
  self.yDiff = torch.Tensor()
end

function TVCriterion:updateOutput(input)
  self.output = input

  return self.output
end

function TVCriterion:updateGradInput(input, gradOutput)
  if self.enabled then
    local C, H, W, batchSize
  
    self.gradInput:resizeAs(input):zero()

    if input:nDimension() == 3 then
      C, H, W = input:size(1), input:size(2), input:size(3)
      self.xDiff:resize(3, H - 1, W - 1)
      self.yDiff:resize(3, H - 1, W - 1)
      self.xDiff:copy(input[{{}, {1, -2}, {1, -2}}])
      self.xDiff:add(-1, input[{{}, {1, -2}, {2, -1}}])
      self.yDiff:copy(input[{{}, {1, -2}, {1, -2}}])
      self.yDiff:add(-1, input[{{}, {2, -1}, {1, -2}}])
      self.gradInput[{{}, {1, -2}, {1, -2}}]:add(self.xDiff):add(self.yDiff)
      self.gradInput[{{}, {1, -2}, {2, -1}}]:add(-1, self.xDiff)
      self.gradInput[{{}, {2, -1}, {1, -2}}]:add(-1, self.yDiff)
    else
      C, H, W = input:size(2), input:size(3), input:size(4)
      batchSize = input:size(1)
      self.xDiff:resize(batchSize, 3, H - 1, W - 1)
      self.yDiff:resize(batchSize, 3, H - 1, W - 1)
      self.xDiff:copy(input[{{}, {}, {1, -2}, {1, -2}}])
      self.xDiff:add(-1, input[{{}, {}, {1, -2}, {2, -1}}])
      self.yDiff:copy(input[{{}, {}, {1, -2}, {1, -2}}])
      self.yDiff:add(-1, input[{{}, {}, {2, -1}, {1, -2}}])
      self.gradInput[{{}, {}, {1, -2}, {1, -2}}]:add(self.xDiff):add(self.yDiff)
      self.gradInput[{{}, {}, {1, -2}, {2, -1}}]:add(-1, self.xDiff)
      self.gradInput[{{}, {}, {2, -1}, {1, -2}}]:add(-1, self.yDiff)

      if self.sizeAverage then
        self.gradInput:div(batchSize)
      end
    end

    self.gradInput:mul(self.strength)
    self.gradInput:add(gradOutput)
  else
    self.gradInput = gradOutput
  end

  return self.gradInput
end

return nn.TVCriterion