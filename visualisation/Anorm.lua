-- Alpha_Norm
-- [output] forward(input) -- updateOutput(input) rather than override forward()
-- [gradInput] backward(input, gradOutput) -- call two functions to compute two gradients here:
		-- updateGradInput(input, gradOutput)
		-- accGradParameters(intput, gradOutput)
		-- rewrite the above two funcs rather than override backward() 
local Anorm, parent = torch.class('nn.Anorm', 'nn.Module')

function Anorm:__init(alpha)
   parent.__init(self)
   self.alpha = alpha or 6
end

function Anorm:updateOutput(input)
	--self.output = self.output
	self.buffer = self.buffer or input.new()
	self.normp = self.normp or input.new()
	self.norm = self.norm or input.new()
	if self.alpha % 2 ~= 0 then
		self.buffer:asb(input):pow(self.alpha)
	else
		self.buffer:pow(input, self.alpha)
	end
	self.normp:sum(self.buffer,2)
    	self.norm:pow(self.normp,1/self.p)
    	return self.norm
end

function Anorm:updateGradInput(input)
	self.gradIn = torch.sign(input):cmul((input:div(self.norm)):pow(self.alpha - 1))
	return self.gradIn
end