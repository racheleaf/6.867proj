import torch
from torch.autograd import Variable
from torchconfig import config


def generate(model, num_chars, seed, temp, char_encoder, char_decoder):
    print('Generating with seed:', seed)
    hidden = model.init_hidden(1)
    result = [seed]
    cur_char = Variable(torch.LongTensor([char_encoder[seed]]))
    if config['cuda']:
        hidden = (hidden[0].cuda(), hidden[1].cuda())
        cur_char = cur_char.cuda()
    print(cur_char.size())
    for i in range(num_chars):
        output, hidden = model(cur_char, hidden)
        distribution = output.view(-1).div(temp).exp()
        sample = torch.multinomial(distribution, 1)
        result.append(char_decoder[sample.data[0]])
        cur_char[:] = sample
    return ''.join(result)
