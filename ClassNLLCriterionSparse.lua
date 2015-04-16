local ClassNLLCriterionSparse, parent = torch.class('nn.ClassNLLCriterionSparse', 'nn.Criterion')

function ClassNLLCriterionSparse:__init(weights)
   parent.__init(self)
   self.sizeAverage = true
   self.outputTensor = torch.Tensor(1)
   if weights then
       assert(weights:dim() == 1, "weights input should be 1-D Tensor")
       self.weights = weights
   end
   -- will be set during updateOutput
   self.total = 0
end

function ClassNLLCriterionSparse:updateOutput(input, target)

   if input:dim() == 1 then
      self.output = -input[target]
      if self.weights then
          self.output = self.output*self.weights[target]
      end
   elseif input:dim() == 2 then
      local output = 0
      for i=1,target:size(1) do
         if target[i] ~= 0 then
            self.total = self.total + 1
            if self.weights then
               output = output - input[i][target[i]]*self.weights[target[i]]
            else
               output = output - input[i][target[i]]
            end
         end
      end
      print('non zero classes: ', self.total)
      if self.sizeAverage then
         output = output / self.total
      end
      self.output = output
   else
      error('matrix or vector expected')
   end
   return self.output
end

function ClassNLLCriterionSparse:updateGradInput(input, target)
   self.gradInput:resizeAs(input)
   self.gradInput:zero()

  if input:dim() == 1 then
      self.gradInput[target] = -1
      if self.weights then
          self.gradInput[target] = self.gradInput[target]*self.weights[target]
      end
  else
     local z = -1
     if self.sizeAverage then
        print("using total: ", self.total)
        z = z / self.total
     end
     for i=1,target:size(1) do
        if target[i] ~= 0 then
           self.gradInput[i][target[i]] = z
           if self.weights then
              self.gradInput[i][target[i]] = self.gradInput[i][target[i]]*self.weights[target[i]]
           end
        end
     end
  end
  return self.gradInput
end
