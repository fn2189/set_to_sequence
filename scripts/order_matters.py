import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt
import seaborn
import pdb


class ReadLinear(nn.Module):
    """
    A read block from the Order Matters architechture. In the case of digits reordering, a small multilayer perceptron
    implemented as 1d conv. Specifically, if the input is of shape (batch size, set_length, input_dim), conv1d with
    1x1 kernel size and F output filters will give us an output shape of (batch size, set_length, F)
    
    Paramters
    ---------
    hidden_dim: size of the digit embedding
    """
    def __init__(self, hidden_dim, input_dim=1):
        super(ReadLinear, self).__init__()
        self.W = nn.Parameter(torch.randn(hidden_dim, input_dim))
        self.b = nn.Parameter(torch.randn(hidden_dim))
        self.nonlinearity = nn.ReLU6()
        
    def forward(self, x):
        """
        x is a batch of sets of shape (batch size, input_dim, set_length) to fit the expected shape of conv1d
        """
        print( f'x shape: {x.size()}')
        W = self.W.unsqueeze(0).unsqueeze(0) #final shape (1, 1, input_dim, output_dim)
        b = self.b.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
        x = x.permute(0,2,1).unsqueeze(-1) #shape (batch size, set_length, input_dim, 1)
        
        x = self.nonlinearity(torch.matmul(W, x)  + b) # shape (batch size, set_length, hidden_dim, 1)
        x = x.squeeze(-1).permute(0,2,1) # shape (batch size, hidden_dim, set_length)
        
        return x

class ReadWordEncoder(nn.Module):
    """
    A read block from the Order Matters architechture. In the character level word encoding, a small multilayer perceptron
    implemented as 1d conv. Specifically, the input is of shape (batch size, set_length, max_word_length, input_size). 
    
    Paramters
    ---------
    hidden_dim: size of the digit embedding
    input_size: character level vocab_size. Default to 26
    """
    
    def __init__(self, hidden_dim, input_size=26):
        super(ReadWordEncoder, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        
    def forward(self, x):
        """
        x is of shape (batch_size, n_set, max_word_length, vocab_size)
        we need to loop over the batch size because lstm batch 1st take input (batch, seq_length, vocab_size)
        and so for each element of the batch we have batch -> n_set, seq_length -> max_word_length, vocab_size -> vocab_size
        """
        #print(f'X[i,:,:,:] shape: {x[0, :, :, :].size()}')
        l = []
        for i in range(x.size(0)):
            outputs, (h_n, c_n) =  self.lstm(x[i, :, :, :])
            l.append(h_n)
        res = torch.cat(l, dim=0).permute(0,2,1) #shape (batch_size, hidden_dim, n_set)
        return res
                     
    
class Process(nn.Module):
    """
    A Process block from the Order Matters architechture. Implemented via a self attention mechanism where in order 
    to compute the next state, we run r_t the attention vector as input for the next step.
    """
    def __init__(self, input_dim, hidden_dim, lstm_steps, batch_size):
        """
        """
        super(Process, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lstm_steps = lstm_steps
        self.batch_size = batch_size
        self.lstmcell = nn.LSTMCell(self.input_dim, self.hidden_dim, bias=True)
        ##QUESTION: Should these be initialized to the same value for each member of the batch ?
        ### TODO: look into how to initialize LSTM state/output
        self.i0 = nn.Parameter(torch.zeros(self.input_dim), requires_grad=False)
        self.h_0 = nn.Parameter(torch.randn(self.hidden_dim), requires_grad=False)
        self.c_0 = nn.Parameter(torch.randn(self.hidden_dim), requires_grad=False)
        
        
    def forward(self, M, mask=None, dropout=None):
        """
        c_t is the state the LSTM evolves, aka q_t from the order matters paper
        h and c are initialized randomly
        the dot product is scaled to avoid it exploding with the embedding dimension
        
        The out put, q_t_star = (q_t, r_t) is the linear  is projected with a linear layer to the size of the state of the write LSTM, and used as its initial state
        
        Parameters
        ----------
        M: the memories tensor or shape ((batch size, hidden_dim, set_length))
        """
        #To account for the last batch that might not have the same length as the rest
        batch_size = M.size(0)
        i0 = self.i0.unsqueeze(0).expand(batch_size, -1)
        h_0 = self.h_0.unsqueeze(0).expand(batch_size, -1)
        c_0 = self.c_0.unsqueeze(0).expand(batch_size, -1)
        
        for _ in range(self.lstm_steps):
            if _ == 0:
                h_t_1 = h_0
                c_t_1 = c_0
                r_t_1 = i0
            h_t, c_t = self.lstmcell(r_t_1, (h_t_1, c_t_1))
            d_k = c_t.size(-1)
            
            #c_t is of shape (batch_size, hidden_dim) so we expand it
            #try:
            scores = torch.matmul(M.transpose(-2, -1), c_t.unsqueeze(2)) \
                         / math.sqrt(d_k)
            #except:
            #    print(f'M: {M.transpose(-2, -1).size()}, c_t: {c_t.size()}')
            #    raise RuntimeError('Score error')
                
            if mask is not None:
                scores = scores.masked_fill(mask == 0, -1e9)
            p_attn = F.softmax(scores, dim = -1)
            if dropout is not None:
                p_attn = dropout(p_attn)
            r_t_1 = torch.matmul(M, p_attn).squeeze(-1)
            #print(f'r_t_1: {r_t_1.size()}')
            h_t_1 = h_t
            c_t_1 = c_t
        return (r_t_1, c_t_1)
    
class Attention(nn.Module):
    """
    Attention model for Pointer-Net taken from https://github.com/shirgur/PointerNet/blob/master/PointerNet.py
    """

    def __init__(self, input_dim,
                 hidden_dim):
        """
        Initiate Attention
        :param int input_dim: Input's diamention
        :param int hidden_dim: Number of hidden units in the attention
        """

        super(Attention, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.input_linear = nn.Linear(input_dim, hidden_dim)
        self.context_linear = nn.Conv1d(input_dim, hidden_dim, 1, 1)
        self.V = nn.Parameter(torch.FloatTensor(hidden_dim), requires_grad=True)
        self._inf = nn.Parameter(torch.FloatTensor([float('-inf')]), requires_grad=False)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

        # Initialize vector V
        nn.init.uniform_(self.V, -1, 1)

    def forward(self, input,
                context,
                mask):
        """
        Attention - Forward-pass
        :param Tensor input: Hidden state h (as said in the Pointer's Network paper:  For the LSTM RNNs, 
        we use the state after the output gate has been component-wise multiplied by the cell activations.
        
        :param Tensor context: Attention context
        :param ByteTensor mask: Selection mask
        
        :return: tuple of - (Attentioned hidden state, Alphas)
        """

        # input is of shape (batch, hidden_dim) so inp will be of shape (batch_size, hidden_dim, seq_len)
        inp = self.input_linear(input.unsqueeze(2).transpose(-2, -1)).transpose(-2, -1).repeat(1,1,context.size(-1))

        # context is shape (batch, hidden_dim, seq_len)
        ctx = self.context_linear(context)

        # V will of shape (batch, 1, hidden_dim)
        V = self.V.unsqueeze(0).expand(context.size(0), -1).unsqueeze(1)

        # attention will be of shape (batch, seq_len)
        att = torch.bmm(V, self.tanh(inp + ctx)).squeeze(1)
        if len(att[mask]) > 0:
            att[mask] = self.inf[mask]
        
        alpha = self.softmax(att)

        hidden_state = torch.bmm(ctx, alpha.unsqueeze(2)).squeeze(2)

        return hidden_state, alpha

    def init_inf(self, mask_size):
        self.inf = self._inf.unsqueeze(1).expand(*mask_size)

        
class Write(nn.Module):
    """
    A Write block from the Order Matters architechture. 
    """
    
    def __init__(self, embedding_dim,
                 hidden_dim):
        """
        Initiate Decoder
        :param int embedding_dim: Number of embeddings in Pointer-Net
        :param int hidden_dim: Number of hidden units for the decoder's RNN
        """

        super(Write, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.input_to_hidden = nn.Linear(embedding_dim, 4 * hidden_dim)
        self.hidden_to_hidden = nn.Linear(hidden_dim, 4 * hidden_dim)
        self.hidden_out = nn.Linear(hidden_dim * 2, hidden_dim)
        self.att = Attention(hidden_dim, hidden_dim)

        # Used for propagating .cuda() command
        self.mask = nn.Parameter(torch.ones(1), requires_grad=False)
        self.runner = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, embedded_inputs,
                decoder_input,
                hidden,
                context):
        """
        Decoder - Forward-pass
        :param Tensor embedded_inputs: Embedded inputs of Pointer-Net
        :param Tensor decoder_input: First decoder's input
        :param Tensor hidden: First decoder's hidden states
        :param Tensor context: Encoder's outputs
        :return: (Output probabilities, Pointers indices), last hidden state
        """
        batch_size = embedded_inputs.size(0)
        # The size of the set
        input_length = embedded_inputs.size(2)

        # (batch, seq_len)
        mask = self.mask.repeat(input_length).unsqueeze(0).repeat(batch_size, 1)
        self.att.init_inf(mask.size())

        # Generating arang(input_length), broadcasted across batch_size
        runner = self.runner.repeat(input_length)
        for i in range(input_length):
            runner.data[i] = i
        runner = runner.unsqueeze(0).expand(batch_size, -1).long()

        outputs = []
        pointers = []

        def step(x, hidden):
            """
            Recurrence step function
            :param Tensor x: Input at time t
            :param tuple(Tensor, Tensor) hidden: Hidden states at time t-1
            :return: Hidden states at time t (h, c), Attention probabilities (Alpha)
            """

            # Regular LSTM
            h, c = hidden #shapes ((batch_size, hidden_dim), (batch_size, hidden_dim))
            #print(f'h shape: {h.size()}')
            #print(f'x shape: {x.size()}')
            
            #gates = self.input_to_hidden(x) + self.hidden_to_hidden(h.squeeze())
            gates = self.hidden_to_hidden(h.squeeze())
            input, forget, cell, out = gates.chunk(4, 1)

            input = torch.sigmoid(input)
            forget = torch.sigmoid(forget)
            cell = torch.tanh(cell)
            out = torch.sigmoid(out)

            c_t = (forget * c) + (input * cell)
            h_t = out * torch.tanh(c_t)
            #print(f'out: {out.size()}, c_t: {c_t.size()}, h_t: {h_t.size()}')

            # Attention section
            hidden_t, output = self.att(h_t, context, torch.eq(mask, 0))
            hidden_t = torch.tanh(self.hidden_out(torch.cat((hidden_t, h_t), 1)))

            return hidden_t, c_t, output

        # Recurrence loop
        for _ in range(input_length):
            h_t, c_t, outs = step(decoder_input, hidden)
            hidden = (h_t, c_t)
            # Masking selected inputs
            masked_outs = outs * mask

            # Get maximum probabilities and indices
            max_probs, indices = masked_outs.max(1)
            one_hot_pointers = (runner == indices.unsqueeze(1).expand(-1, outs.size()[1])).float()

            # Update mask to ignore seen indices
            mask  = mask * (1 - one_hot_pointers)

            # Get embedded inputs by max indices
            #embedding_mask = one_hot_pointers.unsqueeze(2).expand(-1, -1, self.embedding_dim).byte()
            embedding_mask = one_hot_pointers.unsqueeze(1).expand(-1, self.embedding_dim, -1).byte()
            decoder_input = embedded_inputs[embedding_mask.data].view(batch_size, self.embedding_dim)

            outputs.append(outs.unsqueeze(0))
            pointers.append(indices.unsqueeze(1))

        outputs = torch.cat(outputs).permute(1, 0, 2)
        pointers = torch.cat(pointers, 1)

        return outputs, pointers, hidden

    
class ReadProcessWrite(nn.Module):
    """
    The full read-process-write from the order matters paper.
    """
    def __init__(self, hidden_dim, lstm_steps, batch_size, input_dim=1, reader='linear'):
        super(ReadProcessWrite, self).__init__()
        self.readers_dict = {'linear': ReadLinear, 'words': ReadWordEncoder, 'videos': ReadLinear}
        
        #print(f'hidden_dim: {hidden_dim}, input_dim: {input_dim}')
        self.decoder_input0 = nn.Parameter(torch.zeros(hidden_dim))
        self.read = self.readers_dict[reader](hidden_dim, input_dim)
        self.process = Process(hidden_dim, hidden_dim, lstm_steps, batch_size)
        self.write = Write(hidden_dim, hidden_dim)
        self.batch_size = batch_size
        self.process_to_write = nn.Linear(hidden_dim * 2, hidden_dim) #linear layer to project q_t_star to the hidden size of the write block
        
    def forward(self, x):
        batch_size = x.size(0)
        M = self.read(x)
        r_t, c_t = self.process(M)
        q_t_star = torch.cat([r_t, c_t], dim=-1) #shape (batch_size, 2*hidden_dim)
        #print(f'q_t_star shape: {q_t_star.size()}')
        
        #We project q_t_star using a linear layer to the hidden size of the write block to be the initial hidden state
        write_block_hidden_state_0 = self.process_to_write(q_t_star) #shape (batch_size, hidden_dim)
        write_block_output_state_0 = self.decoder_input0.unsqueeze(0).expand(batch_size, -1) #shape (batch_size, hidden_dim)
        decoder_input0 = self.decoder_input0.unsqueeze(0).expand(batch_size, -1) #shape (batch_size, hidden_dim)
        
        #print('decoder_input0: ', decoder_input0)
        decoder_hidden0 = (write_block_output_state_0, write_block_hidden_state_0)
        outputs, pointers, hidden = self.write(M,
                                               decoder_input0,
                                               decoder_hidden0,
                                                 M)
        return outputs, pointers, hidden
    
    
