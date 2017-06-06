local AlphaCriterion, parent = torch.class('nn.AlphaCriterion', 'nn.Module')

function AlphaCriterion:__init(alpha)
  parent.__init(self)
  self.sizeAverage = true
  self.alpha = alpha or 6
end

function AlphaCriterion:updateOutput(input)
  self.output = input
  return self.output
end

function AlphaCriterion:updateGradInput(input, gradOutput)
  self.gradInput:resizeAs(input)
  norm = torch.norm(input, self.alpha)
  self.gradInput:copy(torch.cmul(torch.pow(torch.abs(input):div(norm + 1e-20), self.alpha - 1), torch.sign(input)))
  self.gradInput:add(gradOutput)
  return self.gradInput
end

return nn.AlphaCriterion
