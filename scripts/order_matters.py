import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt
import seaborn


class Read(nn.Module):
    """
    A read block from the Order Matters architechture. In the case of digits reordering, a small multilayer perceptron
    implemented as 1d conv. Specifically, if the input is of shape (batch size, set_length, input_dim), conv1d with
    1x1 kernel size and F output filters will give us an output shape of (batch size, set_length, F)
    
    Paramters
    ---------
    hidden_dim: size of the digit embedding
    """
    def __init__(self, hidden_dim, input_dim=1):
        super(Read, self).__init__()
        self.conv1d = nn.Conv1d(input_dim, hidden_dim, 1, 1, bias=True)
        
    def forward(self, x):
        """
        x is a batch of sets of shape (batch size, input_dim, set_length) to fit the expected shape of conv1d
        """
        x = self.conv1d(x)
        #print(x.size())
        return x
    
    
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
        self.i0 = torch.zeros(self.input_dim)
        
        
    def forward(self, M, mask=None, dropout=None):
        """
        c_t is the state the LSTM evolves, aka q_t from the order matters paper
        h and c are initialized randomly
        the dot product is scaled to avoid it exploding with the embedding dimension
        
        Parameters
        ----------
        M: the memories tensor or shape ((batch size, hidden_dim, set_length))
        """
        h_0 = torch.randn(self.batch_size, self.hidden_dim)
        c_0 = torch.randn(self.batch_size, self.hidden_dim)
        i0 = self.i0.expand(self.batch_size, -1)
        for _ in range(self.lstm_steps):
            if _ == 0:
                h_t_1 = h_0
                c_t_1 = c_0
                r_t_1 = i0
            h_t, c_t = self.lstmcell(i0, (h_t_1, c_t_1))
            ## We accumulate the cell state at each step
            d_k = c_t.size(-1)
            
            #c_t is of shape (batch_size, hidden_dim) so we expand it 
            scores = torch.matmul(M.transpose(-2, -1), c_t.unsqueeze(2)) \
                     / math.sqrt(d_k)
                
            if mask is not None:
                scores = scores.masked_fill(mask == 0, -1e9)
            p_attn = F.softmax(scores, dim = -1)
            if dropout is not None:
                p_attn = dropout(p_attn)
            r_t_1 = torch.matmul(M, p_attn)
            h_t_1 = h_t
            c_t_1 = c_t
        return (r_t_1, c_t_1)
    
class Attention(nn.Module):
    """
    Attention model for Pointer-Net
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
        self.softmax = nn.Softmax()

        # Initialize vector V
        nn.init.uniform(self.V, -1, 1)

    def forward(self, input,
                context,
                mask):
        """
        Attention - Forward-pass
        :param Tensor input: Hidden state h
        :param Tensor context: Attention context
        :param ByteTensor mask: Selection mask
        :return: tuple of - (Attentioned hidden state, Alphas)
        """

        # (batch, hidden_dim, seq_len)
        #print('input: ', input.unsqueeze(2).transpose(-2, -1))
        inp = self.input_linear(input.unsqueeze(2).transpose(-2, -1)).transpose(-2, -1).repeat(1,1,5)

        # context is shape (batch, seq_len, hidden_dim)
        ctx = self.context_linear(context)

        # (batch, 1, hidden_dim)
        V = self.V.unsqueeze(0).expand(context.size(0), -1).unsqueeze(1)

        # (batch, seq_len)
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
            h, c = hidden
            gates = self.input_to_hidden(x) + self.hidden_to_hidden(h.squeeze())
            input, forget, cell, out = gates.chunk(4, 1)

            input = torch.sigmoid(input)
            forget = torch.sigmoid(forget)
            cell = torch.tanh(cell)
            out = torch.sigmoid(out)

            c_t = (forget * c) + (input * cell)
            h_t = out * torch.tanh(c_t)

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
    def __init__(self, hidden_dim, lstm_steps, batch_size, input_dim=1):
        super(ReadProcessWrite, self).__init__()
        #self.decoder_input0 = nn.Parameter(torch.FloatTensor(hidden_dim), requires_grad=False)
        self.decoder_input0 = torch.zeros(hidden_dim)
        self.read = Read(hidden_dim, input_dim)
        self.process = Process(hidden_dim, hidden_dim, lstm_steps, batch_size)
        self.write = Write(hidden_dim, hidden_dim)
        
    def forward(self, x):
        M = self.read(x)
        r_t, c_t = self.process(M)
        decoder_input0 = self.decoder_input0.unsqueeze(0).expand(batch_size, -1)
        #print('decoder_input0: ', decoder_input0)
        decoder_hidden0 = (r_t, c_t)
        outputs, pointers, hidden = self.write(M,
                                               decoder_input0,
                                               decoder_hidden0,
                                                 M)
        return outputs, pointers, hidden
    
    
