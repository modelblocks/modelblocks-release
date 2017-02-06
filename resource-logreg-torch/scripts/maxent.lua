require 'cutorch'
require 'cunn'
utilx = require 'pl.utils'
tds = require 'tds'
require 'math'
require 'optim'
require 'torch'

cmd = torch.CmdLine()
cmd:text()
cmd:text('MaxEnt in Torch with GPU')
cmd:text('Options:')
-- argument, default value, explanation
cmd:option('-weightdecay', 1e-3, 'lambda for L2 regularization/weight decay of the linear layer')
cmd:option('-learningrate', 1e-1, 'learning rate for SGD')
cmd:option('-learningratedecay', 1e-4, 'learning rate decay for SGD')
cmd:option('-batchsize', 50000, 'samples in a single batch')
cmd:option('-lossdifference', 1e-4, 'the absolute loss difference between two batches')
cmd:option('-trainingdata', 'filename', 'file name of the training preds file')
cmd:option('-outputfile', 'outputfilename', 'file name of the output parameters file')
cmd:option('-optimizer', 'adam', 'optimizer used in optimization')
cmd:option('-accuracyswitch', 1, 'switch for displaying accuracy for each epoch')
cmd:text()
opt = cmd:parse(arg)

xlabels = tds.Vec()
ylabels = tds.Vec()
xlabel_ds = tds.Hash()
ylabel_ds = tds.Hash()
xs = {}
ys = {}

-- data loading
print('Reading in data...')
for line in io.lines(opt.trainingdata) do
    local xline = utilx.split(line, ' : ')
    local fs = utilx.split(xline[1], ',')
    local f_vector = {}
    for i = 1, #fs do
        if xlabel_ds[fs[i]] == nil then
            xlabel_ds[fs[i]] = #xlabel_ds + 1
            xlabels:insert(fs[i])
            f_vector[i] = {xlabel_ds[fs[i]], 1}
        else
            f_vector[i] = {xlabel_ds[fs[i]], 1}
        end
    end
    if ylabel_ds[xline[2]] == nil then
        ylabel_ds[xline[2]] = #ylabel_ds + 1
        ylabels:insert(xline[2])
    end
    xs[#xs + 1] = torch.Tensor(f_vector):cuda()
    ys[#ys + 1] = ylabel_ds[xline[2]]
end

ys_tensor = torch.Tensor(ys):cuda()
-- model
print('Preparing the model...')
sparselinear = nn.SparseLinear(#xlabels, #ylabels)
model = nn.Sequential():add(sparselinear):add(nn.LogSoftMax())
model:cuda()
params, grad_params = model:getParameters()
obj = nn.ClassNLLCriterion()
obj:cuda()

--feval
this_x = 0
this_y = 0
acc = 0
function feval(new_params)
    if params ~= new_params then
        params:clone(new_params)
    end
    grad_params:zero()
    local output = model:forward(this_x)
    -- print(output)
    if opt.accuracyswitch == 0 then
        goto switchoff
    end
    _, cor_index = torch.max(output, 2)
    assert((#cor_index)[1] == (#this_y)[1], (#cor_index)[1])
    -- print(cor_index, this_y)
    for i = 1, (#cor_index)[1] do
        if cor_index[{i,1}] == this_y[i] then
            acc = acc + 1
        end
    end
    ::switchoff::
    local loss = obj:forward(output, this_y)
    local grad_loss = obj:backward(output, this_y)
    model:backward(this_x, grad_loss)
    return loss, grad_params
end

-- training
hyperparams = {}
hyperparams.learningRate = opt.learningrate
hyperparams.weightDecay = opt.weightdecay
hyperparams.learningRateDecay = opt.learningratedecay
batchsize = opt.batchsize
lossdifference = opt.lossdifference
cur_index = 1
data_size = #xs
prev_loss = math.huge
epoch_iter = math.ceil(#xs / opt.batchsize)
print('Start training...')
print('Each epoch is ' .. epoch_iter .. ' iterations.')
cum_loss = 0
iter_counter = 0
while true do
    iter_counter = iter_counter + 1
    local x_begin = cur_index
    local x_end = cur_index + batchsize - 1
    if x_end > data_size then
        x_end = data_size
    end
    this_x = {}
    for i = x_begin, x_end do
        this_x[#this_x + 1] = xs[i]
    end
    this_y = ys_tensor:sub(x_begin, x_end)
--    print(this_x, this_y)
    if opt.optimizer == 'sgd' then
        _, fx = optim.sgd(feval, params, hyperparams)
    elseif opt.optimizer == 'adam' then
        _, fx = optim.adam(feval, params, hyperparams)
    else
        error('unknown optimizer!') 
    end
    cum_loss = cum_loss + fx[1]
--    print('Iteration: ', hyperparams.t, '; Loss: ', string.format("%.4f", fx[1]))
    if iter_counter % epoch_iter == 0 then
        if opt.accuracyswitch == 1 then
            print('Epoch: ', iter_counter / epoch_iter, '; Loss: ', string.format("%.4f",cum_loss / epoch_iter), '; Acc: ', string.format('%.4f',acc/ data_size))
        else
            print('Epoch: ', iter_counter / epoch_iter, '; Loss: ', string.format("%.4f",cum_loss / epoch_iter))
        end
        if math.abs( prev_loss - cum_loss ) < opt.lossdifference then
            break
        else
            prev_loss = cum_loss
            cum_loss = 0
            acc = 0
        end
    end
    cur_index = x_end
    if cur_index == data_size then
        cur_index = 1
    end
end

-- output weights
print('Writing output...')
weights = sparselinear.weight
bias = sparselinear.bias
output_file_handle = io.open(opt.outputfile, 'w')
for i = 1, #ylabels do -- bias
    output_file_handle:write(' : ' .. ylabels[i] .. ' = ' .. string.format("%.8f", bias[i]) .. '\n')
end
for i = 1, #xlabels do -- weights
    for j = 1, #ylabels do
        output_file_handle:write(string.sub(xlabels[i],1,-3) .. ' : ' .. ylabels[j] .. ' = ' .. string.format("%.8f", weights[{j, i}]) .. '\n')
    end
end
output_file_handle:close()
