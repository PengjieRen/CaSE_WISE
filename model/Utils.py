import torch
import torch.nn.functional as F

def build_map(map_sequence, vocab_size=None):
    batch_size, seq_len = map_sequence.size()
    if torch.cuda.is_available():
        b_map_ = torch.cuda.FloatTensor(batch_size, seq_len, vocab_size).fill_(0)
    else:
        b_map_ = torch.zeros(batch_size, seq_len, vocab_size)
    b_map_.scatter_(2, map_sequence.unsqueeze(2), 1.)
    b_map_.requires_grad=False
    return b_map_

def pairwise_binary_cross_entropy_with_logits(x, y, reduction='mean'):
    x=x.unsqueeze(-1).expand(-1, -1, x.size(1))
    y=y.unsqueeze(-1).expand(-1, -1, y.size(1))
    x= x - x.transpose(1, 2)
    y= y - y.transpose(1, 2)

    return F.soft_margin_loss(x, y, reduction=reduction)

NEAR_INF = 1e20
NEAR_INF_FP16 = 65504
def neginf(dtype):
    """Return a representable finite number near -inf for a dtype."""
    if dtype is torch.float16:
        return -NEAR_INF_FP16
    else:
        return -NEAR_INF

def generate_square_subsequent_mask(sz, random=False):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(~mask, neginf(torch.float32)).masked_fill(mask, float(0.0))
    if torch.cuda.is_available():
        mask = mask.cuda()
    if random:
        mask.masked_fill_(torch.randn_like(mask)>0.8, neginf(torch.float32))
    return mask
