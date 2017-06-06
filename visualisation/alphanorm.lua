-- Alpha_Norm
-- [output] forward(input) -- updateOutput(input) rather than override forward()
-- [gradInput] backward(input, gradOutput) -- call two functions to compute two gradients here:
		-- updateGradInput(input, gradOutput)
		-- accGradParameters(intput, gradOutput)
		-- rewrite the above two funcs rather than override backward() 
local alphanorm, parent = torch.class('nn.alphanorm', 'nn.Module')

function alphanorm:__init(alpha)
   parent.__init(self)
   self.alpha =  alpha or 6
end

function alphanorm:updateOutput(input)
	self.output = input
	return self.output
end

function alphanorm:updateGradInput(input, gradOutput)
    self.gradInput:resizeAs(input)
    norm = torch.norm(input, self.alpha)
    self.gradInput:copy(torch.cmul(torch.pow(torch.abs(input):div(norm+1e-20), self.alpha - 1),torch.sign(input)))
    self.gradInput:add(gradOutput)
    return self.gradInput
end

return nn.alphanorm