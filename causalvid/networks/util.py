import torch 

def length_to_mask(length, max_len=None, dtype=None):
    """length: B.
    return B x max_len.
    If max_len is None, then max of length will be used.
    lens = [3, 5, 4]
    mask = [[1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 0]]
    """
    assert len(length.shape) == 1, 'Length shape should be 1 dimensional.'
    max_len = max_len or length.max().item()
    mask = torch.arange(max_len, device=length.device,
                        dtype=length.dtype).expand(len(length), max_len) < length.unsqueeze(1)
    if dtype is not None:
        mask = torch.as_tensor(mask, dtype=dtype, device=length.device)
    return mask

def get_mask(lengths, max_length):
    """ Computes a batch of padding masks given batched lengths """
    mask = 1 * (
        torch.arange(max_length).unsqueeze(1).to(lengths.device) < lengths
    ).transpose(0, 1)
    return mask

    
def sample_bg(vis_feat):
    ''' in batch sample, each video is composed with frames from other video 
        in/out: [bs v_len, dim]
    '''
    bs, v_len, dim = vis_feat.size()
    weight = 1-torch.eye(bs).to(vis_feat.device)
    weight = torch.repeat_interleave(weight, v_len, dim=1)
    sample_idx = torch.multinomial(weight, v_len, False).long()
    sample_bg = vis_feat.view(-1, dim)[sample_idx]
    return sample_bg


def left_stack(vid_feats, fg_mask):
    """
    assemble value to left, [1,0,2,0,3]-->[1,2,3,0,0], pad with 0
    vid_feats: (bs, 16, 4096)
    fg_mask: (bs, 16,) float mask
    """

    pad = vid_feats.new_ones(vid_feats.size(0),1,vid_feats.size(-1))
    vid_feats = torch.cat([vid_feats, pad], dim=1)

    bs, v_len = fg_mask.size()
    tmp = torch.arange(1,v_len+1).expand(bs, -1).to(vid_feats.device)
    tmp = fg_mask*tmp
    tmp = (tmp -1).long()
    tmp = torch.where(tmp!=-1, tmp, v_len)
    tmp, _ = torch.sort(tmp, dim=1)
    vid_feats = vid_feats[torch.arange(bs).unsqueeze(-1), tmp]

    # # get idx for unstack
    # idx_fg = torch.arange(-v_len, 0).expand(bs, -1).to(vid_feats.device)
    # idx_bg = torch.arange(0, v_len).expand(bs, -1).to(vid_feats.device)

    # idx =  torch.where(fg_mask.bool(), idx_fg, idx_bg).long()
    # idx, _ = torch.sort(idx, dim=1)
    # idx = torch. where(idx<0, idx+v_len, idx)  

    return vid_feats