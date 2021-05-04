import numpy as np
import torch
from torch import nn
from functools import wraps
import time
import torch.autograd.profiler as profiler
from torch import multiprocessing
from fairmotion.models.transformer import PositionalEncoding

TIMING_LOGS_VERBOSITY_LEVEL = 19  # all logs >= this verbosity will print
RUNTIMES = []  # this is a bad idea, but I'm doing it anyway because it makes sorting convenient

def get_runtime(classname=None, verbosity_level=5):
    def get_runtime_noarg(func):
        @wraps(func)
        def wrapped(*args, **kwargs):
            if verbosity_level < TIMING_LOGS_VERBOSITY_LEVEL:
                return func(*args, **kwargs)
            start = time.time()
            ret = func(*args, **kwargs)
            end = time.time()
            func_name = func.__name__
            if classname is not None:
                func_name = classname + '.' + func_name
            RUNTIMES.append((end-start, func_name))
            # print('"{}" took {:.4f} secs to execute\n'.format(func_name, (end - start)))
            return ret
        return wrapped
    return get_runtime_noarg

def convert_joints_from_3d_to_4d(tensor, N,M):
    '''
    input shape: (B, T, N*M) (ie. batch, seq_len, input_dim)
    output shape: (B, T, N, M)
    '''
    return tensor.reshape(tensor.shape[0], tensor.shape[1], tensor.shape[2] // M, tensor.shape[2] // N)

def convert_joints_from_4d_to_3d(tensor):
    '''
    input shape: (B, T, N, M) (ie. batch, seq_len, input_dim)
    output shape: (B, T, N*M)
    '''   
    return tensor.reshape(tensor.shape[0], tensor.shape[1], tensor.shape[2] * tensor.shape[3])

# Make sure to tag this model with 'st_transformer' in training because motion_prediction/utils/prepare_tgt_seqs prepares the correct
# target seq! 
class AutoRegressiveSpatioTemporalTransformer(nn.Module):
    def __init__(self, N, D, M=9, L=1, dropout_rate=0.1, num_heads=4, feedforward_size=256, device=None):
        super(AutoRegressiveSpatioTemporalTransformer, self).__init__()
        self.spatio_temporal_transformer = SpatioTemporalTransformer(N, D, M, L, dropout_rate, num_heads, feedforward_size, device=device)

    def forward(self, src, tgt, max_len=None, teacher_forcing_ratio=None):
        '''
        Only pay attention to src, tgt. max_len and teacher_forcing_ratio are 
        kept in parameter list to adhere to training API.

        src, tgt: (B,T, M*N)
        returns: (B,T, M*N)
        '''
        in_len = src.shape[1]
        out_len = tgt.shape[1]
        full_seq = torch.cat([src, tgt], dim=1)
        '''
        Training works differently than test. In training, we combine source
        and target and predict t+1 at each point. The trainer should know this 
        and coordinate loss accordingly.

        Basically in training we always only predict one timestep in the future,
        but importantly the trainer must do somethig like:
        loss = criterion(src_cat_with_tgt[1:-1], out)
        as opposed to
        loss = criterion(target, out)
        '''
        if self.training:
            # creates T predictions of 1 timestep ahead
            return self.spatio_temporal_transformer(full_seq)
        
        else:
            # we are in test mode, we actually would like to predict out_len
            # frames into the future and compare tgt with our out. The variable
            # "tgt" is never used so as to not cheat (since now tgt is being 
            # used for comparison)
            inputs = torch.zeros(full_seq.shape)
            inputs[:, 0:in_len] = src
            preds_at_timestep = []
            for t in range(out_len):
                pred_one_timestep_ahead = self.spatio_temporal_transformer(inputs[:, t:in_len+t])
                # the last element must go into our input for the next round
                inputs[:, in_len+t] = pred_one_timestep_ahead[:, -1]

            # return the parts that were predicted by the model
            return inputs[:, in_len:]

class SpatioTemporalTransformer(nn.Module):

    def __init__(self, N, D, M=9, L=1, dropout_rate=0.1, num_heads=4,
            feedforward_size=128, input_len=None, pred_len=None, device=None):
        """
        :param N: The number of joints that are in each pose in
        the input.
        :param D: The size of the initial joint embeddings.
        """
        super(SpatioTemporalTransformer, self).__init__()

        print('Params:')
        print(f'N = {N}')
        print(f'D = {D}')
        print(f'M = {M}')
        print(f'L = {L}')
        print(f'num_heads = {num_heads}')
        print(f'feedforward_size = {feedforward_size}')
        print(f'input_len = {input_len}')
        print(f'pred_len = {pred_len}')

        self.device = device
        if self.device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.embedding_layer = JointEmbeddingLayer(N, D, M, self.device)
        self.position_encoding_layer = PositionalEncoding(N*D, dropout_rate)
        self.attention_layers = nn.Sequential(
                                    *[AttentionLayer(D, N, num_heads, dropout_rate, feedforward_size, device=self.device) for _ in range(L)])

        # one last linear layer (not sure what shapes to do yet)
        self.final_linear_layer = nn.Linear(D, M)
        # prediction projection? Need output to be of certain length
        if input_len is not None:
            self.prediction_layer = nn.Linear(input_len, pred_len)

        self.N = N
        self.D = D
        self.M = M
        self.input_len = input_len

    def forward(self, inputs, tgt=None, max_len=None, teacher_forcing_ratio=None):
        embeddings = self.position_encoding_layer(self.embedding_layer(inputs))

        # reverse batch and sequence length for attention layers because
        # nn.MultiheadAttention expects input of (T, B, N*D)
        embeddings = embeddings.permute(1, 0, 2)

        out = self.attention_layers(embeddings)

        out = convert_joints_from_3d_to_4d(out, self.N, self.D)
        out = self.final_linear_layer(out)
        out = convert_joints_from_4d_to_3d(out)

        # Transpose back into (B, T, H)
        out = out.permute(1, 0, 2)
        out.add_(inputs)  # residual layer

        # project to desired output length

        # AutoRegressiveSpatioTemporalTransformer does not want this step
        if self.input_len is not None:
            out = out.permute(0, 2, 1)
            out = self.prediction_layer(out)
            out = out.permute(0, 2, 1)

        return out

    def init_weights(self):
        '''
        No ide if this is right - copied from other models
        '''
        for name, param in self.named_parameters():
            nn.init.uniform_(param.data, -0.08, 0.08)


class JointEmbeddingLayer(nn.Module):
    
    def __init__(self, N, D, M=9, device=None):
        """Transforms joint space M to embedding space D. Each joint has its own weights."""
        super(JointEmbeddingLayer, self).__init__()

        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # I do the W and bias initialization like this to ensure that the weights 
        # are initialized exactly like Pytorch does it.
        linears = [nn.Linear(in_features=M, out_features=D) for _ in range(N)]
        self.W = nn.Parameter(torch.stack([lin.weight for lin in linears]).permute(0, 2, 1).to(device), requires_grad=True)
        self.bias = nn.Parameter(torch.stack([lin.bias for lin in linears]).unsqueeze(0).unsqueeze(0).to(device), requires_grad=True)
        # self.W2 = nn.Parameter(torch.stack([lin.weight for lin in linears]).permute(0, 2, 1), requires_grad=True)
        # self.bias2 = nn.Parameter(torch.stack([lin.bias for lin in linears]).unsqueeze(0).unsqueeze(0), requires_grad=True)
        # print(self.W2.type())
        # print(self.bias2.type())
        # Saving these because they are helpful for reshaping inputs / outputs
        self.M = M
        self.N = N

    def forward(self, inputs):
        """
        input shape: (B, T, N*M) (ie. batch, seq_len, input_dim)
        output shape: (B, T, N*D) 
        """
        # print(inputs.type())
        inputs = convert_joints_from_3d_to_4d(inputs, self.N, self.M)
        out = torch.einsum("btnm,nmd->btnd", inputs, self.W) + self.bias
        return convert_joints_from_4d_to_3d(out)

class AttentionLayer(nn.Module):

    def __init__(self, embed_dim, num_joints, num_heads, dropout_rate=0.1, feedforward_size=256, device=None):
        """The core module with both spatial attention module and 
           temporal attention model embedded within it.
        """
        super(AttentionLayer, self).__init__()
        self.spatial_attention = SpatialAttentionLayer(
                                    embed_dim,
                                    N=num_joints,
                                    H=num_heads,
                                    dropout_rate=dropout_rate,
                                    device=device
                                )
        self.temporal_attention = TemporalAttentionLayer(
                                    embed_dim,
                                    N=num_joints,
                                    H=num_heads,
                                    dropout_rate=dropout_rate,
                                    device=device
                                )

        # two layer feedforward
        self.linear1 = nn.Linear(embed_dim, feedforward_size)
        self.linear2 = nn.Linear(feedforward_size, embed_dim)

        self.layer_norm = nn.LayerNorm(embed_dim*num_joints)
        self.layer_norm_small = nn.LayerNorm(embed_dim)
        self.inplace_dropout = nn.Dropout(dropout_rate, inplace=True)
        self.inplace_relu = nn.ReLU(inplace=True)

        self.N = num_joints
        self.D = embed_dim

    def forward(self, inputs):
        """
        :param inputs: shape (T, B, H)
        :returns out: shape (T, B, H)
        """
        
        # these are obviously not right, just putting this in here as placeholders for now in this form so it 
        # feeds-forward without error
        spatial_out = self.spatial_attention.forward(inputs)
        spatial_out.add_(inputs)  # residual layer
        spatial_out = self.layer_norm(spatial_out)

        temporal_out = self.temporal_attention.forward(inputs)
        temporal_out.add_(inputs)  # residual layer
        temporal_out = self.layer_norm(temporal_out)

        attention_out = spatial_out  # Rename for clear reading, no new allocation
        attention_out.add_(temporal_out)
        attention_out = convert_joints_from_3d_to_4d(attention_out, self.N, self.D)

        out = self.linear1(attention_out)
        self.inplace_relu(out)  # Relu is used here as described in "Attention is All You Need"
        out = self.linear2(out)

        self.inplace_dropout(out)
        out.add_(attention_out)  # residual layer
        out = self.layer_norm_small(out)

        out = convert_joints_from_4d_to_3d(out)

        return out


class SpatialAttentionLayer(nn.Module):
    '''
    K and V are shared across joints
    Q is joint specific
    '''
    def __init__(self, D, H=8, N=20, dropout_rate=0.1, device=None):
        """
        F = D / H = D / 8
        """
        super(SpatialAttentionLayer, self).__init__()
        self.N = N
        self.D = D
        self.F = int(D / H)
        self.device = device
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # These heads are shared across timesteps
        # so below for each timestep T, we are using the same set of "heads"
        self.heads = nn.ModuleList([SpatialAttentionHead(N, D, self.F, self.device) for _ in range(H)])
        self.inplace_dropout = nn.Dropout(dropout_rate, inplace=True)

    def forward(self, inputs):
        '''
        inputs:
            (T, B, N*D)
        Returns:
            (T, B, N*D)
        '''
        T, B, _ = inputs.size()
        inputs = inputs.reshape(T, B, self.N, self.D)
        outputs = torch.zeros(T, B, self.N * self.D).to(self.device)

        for i in range(T):
            timestep_inputs = inputs[i]
            attns = torch.zeros(B, self.N, self.D).to(self.device)
            start = 0
            for head in self.heads:
                attn = head.forward(timestep_inputs)
                attns[:, :, start:start+self.F] = attn
                start += self.F
            # each attn is (B, N, F)
            # flatten --> (B, N*D)
            attns = torch.flatten(attns, start_dim=1)
            outputs[i] = attns
        # dropout
        self.inplace_dropout(outputs)
        return outputs


class SpatialAttentionHead(nn.Module):

    def __init__(self, N, D, F, device=None):
        """One of the heads in the SpatialAttentionLayer
        """
        super(SpatialAttentionHead, self).__init__()
        self.device = device
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.D = D
        self.F = F
        self.sqrt_F = np.sqrt(self.F)
        self.k = nn.Linear(D, F)
        self.v = nn.Linear(D, F)
        # Each joint has its own weights
        self.joint_Qs = nn.ModuleList([nn.Linear(D, F) for _ in range(N)])
        self.softmax = nn.Softmax(dim=2)

    def forward(self, inputs):
        '''
        inputs: (B, N, D)
        Equation (5)

        Each head shall return (N, F)
        '''
        B, N, D = inputs.size()
        k_outputs = self.k(inputs)
        v_outputs = self.v(inputs)
        q_outputs = torch.zeros(B, N, self.F).to(self.device)
        i = 0
        for q in self.joint_Qs:
            q_outputs[:, i, :] = q(inputs[:, i, :])
            i += 1

        attn = torch.matmul(
            q_outputs, k_outputs.transpose(-2, -1)) / self.sqrt_F
        attn = self.softmax(attn)
        # head = A*V (B, N, F)
        attn = torch.matmul(attn, v_outputs)
        return attn


class TemporalAttentionLayer(nn.Module):
    '''
    '''
    def __init__(self, D, H=8, N=20, dropout_rate=0.1, device=None):
        """
        F = D / H = D / 8
        """
        super(TemporalAttentionLayer, self).__init__()
        self.N = N
        self.D = D
        # Each joint uses a separate MHA
        self.MHAs = nn.ModuleList([nn.MultiheadAttention(D, H) for _ in range(N)])
        self.inplace_dropout = nn.Dropout(dropout_rate, inplace=True)
        # Maybe consider passing the "device"
        # variable all the way from train.py
        self.device = device
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def forward(self, inputs):
        '''
        inputs:
            (T, B, N*D)
        Returns:
            (T, B, N*D)
        '''
        T, B, _ = inputs.size()

        inputs = inputs.reshape(T, B, self.N, self.D)
        # mask of dimension (T, D), with 1s in lower triangle
        # and zeros else where
        attn_mask = torch.ones(T, T).to(self.device)
        attn_mask = torch.tril(attn_mask, diagonal=-1)
        outputs = torch.zeros(T, B, self.N * self.D).to(self.device)
        for i in range(self.N):
            start = self.D * i
            # joint_inputs grabs (T, B, D) for joint i
            joint_inputs = inputs[:, :, i, :]
            mha = self.MHAs[i]
            # attn_mask prevents information leak from future time steps
            joint_outputs, joint_outputs_weights = mha(
                joint_inputs, joint_inputs, joint_inputs, attn_mask=attn_mask)
            outputs[:, :, start:start+self.D] = joint_outputs
        # dropout
        self.inplace_dropout(outputs)
        return outputs



# Quick test code for sanity checking
if __name__ == '__main__':

    print(torch.cuda.get_device_name(0))
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    N = 24
    M = 9
    D = 64
    T = 12
    B = 124
    x = torch.rand(B, T, N*M).to(DEVICE)

    model = SpatioTemporalTransformer(N,D, num_heads=4, L=4, feedforward_size=128)
    model = model.to(DEVICE)

    # x = torch.rand(B, N*M, T)

    import time
    start = time.time()

    y = model(x)

    
    print("forward time: ", time.time() - start)
    print(x.shape)
    print(y.shape)  # B, T, N*D
    loss = y.sum()

    start = time.time()
    loss.backward()
    print('backward time', time.time() -start)

    # picking a few modules randomely within the model to ensure 
    # they have grad. ".grad" will give a warning or error if we did something 
    # wrong.
    #print(model.attention_layers[0].linear1.weight.grad.shape)
    print(model.embedding_layer.W.grad.shape)

    param_count = 0
    for parameter in model.parameters():
        param_count += parameter.numel()
    print('param count', param_count)
    
    # print("Q,K,V weights for temporal attention stacked")
    # print(model.attention_layers[0].temporal_attention.in_proj_weight.shape

    RUNTIMES.sort()
    for runtime, name in RUNTIMES:
        print('"{}" took {:.4f} secs to execute'.format(name, runtime))
    

