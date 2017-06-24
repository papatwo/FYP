-- Alpha_Norm
-- [output] forward(input) -- updateOutput(input) rather than override forward()
-- [gradInput] backward(input, gradOutput) -- call two functions to compute two gradients here:
		-- updateGradInput(input, gradOutput)
		-- accGradParameters(intput, gradOutput)
		-- rewrite the above two funcs rather than override backward() 
local atest, parent = torch.class('nn.atest', 'nn.Module')

function atest:__init(alpha)
   parent.__init(self)
   self.alpha =  alpha
end

function atest:updateOutput(input)
	--self.output = self.output
		self.buffer = self.buffer or input.new()
		self.normp = self.normp or input.new()
		self.norm = self.norm or input.new()
		input = input:double()
		--if self.alpha % 2 ~= 0 then
			self.buffer:abs(input):pow(6)
		--else
			--self.buffer:pow(input, self.alpha)
		--end
		self.normp:sum(self.buffer)
		--self.normp:sum(self.normp,1)
	    	self.norm:pow(self.normp,1/6)
	    	return self.norm
end

function atest:updateGradInput(input, gradOutput)
   
		sign = torch.sign(input)
		--n1 = torch.cdiv(input[1], norm)
		n1 = torch.div(input[1], 119.1291)
		n2 = torch.div(input[2], 116.7029)
		n3 = torch.div(input[3], 114.3524)
		pow_n1 = torch.pow(n1, 6)
		pow_n2 = torch.pow(n2, 6)
		pow_n3 = torch.pow(n3, 6)
		gradIn1 = torch.cmul(sign[1], pow_n1)
		gradIn2 = torch.cmul(sign[2], pow_n2)
		gradIn3 = torch.cmul(sign[3], pow_n3)
		all = torch.Tensor(3,224,224)
		all[1] = gradIn1
		all[2] = gradIn2
		all[3] = gradIn3
		self.gradIn = all--torch.cmul(sign, pow_n)
		--self.gradIn = torch.sign(input):cmul((input:div(norm)):pow(6--[[self.alpha]] - 1))
		return self.gradIn
end