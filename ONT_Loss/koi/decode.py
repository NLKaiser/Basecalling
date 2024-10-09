import torch
from koi._runtime import lib, ffi
from koi.utils import void_ptr, empty, zeros


def to_str(x, encoding="ascii"):
    return x[x.nonzero().squeeze(-1)].numpy().tobytes().decode(encoding)


HP_STATES = {
    # fixed stays removed
    64: [0, 21, 42, 63],
    256: [0, 85, 170, 255],
    1024: [0, 341, 682, 1023],
    4096: [0, 1365, 2730, 4095],
}

def beam_search(
    scores,
    beam_width=32,
    beam_cut=100.0,
    scale=1.0,
    offset=0.0,
    blank_score=2.0,
    move_pad=True,
    return_hp_probs=False
):

    if scores.dtype != torch.float16:
        raise TypeError("Expected fp16 but received %s" % scores.dtype)

    assert scores.is_contiguous()

    N, T, C = scores.shape

    chunks = torch.empty((N, 4), device=scores.device, dtype=torch.int32)
    chunks[:, 0] = torch.arange(0, T * N, T)
    chunks[:, 2] = torch.arange(0, T * N, T)
    chunks[:, 1] = T
    chunks[:, 3] = 0
    chunk_results = empty((N, 8), device=scores.device, dtype=torch.int32)

    # todo: reuse scores buffer?
    aux = empty(
        N * (T + 1) * (C + 4 * beam_width), device=scores.device, dtype=torch.int8
    )
    path = zeros(N * (T + 1), device=scores.device, dtype=torch.int32)

    moves = zeros(N * T, device=scores.device, dtype=torch.int8)
    sequence = zeros(N * T, device=scores.device, dtype=torch.int8)
    qstring = zeros(N * T, device=scores.device, dtype=torch.int8)

    args = [
        void_ptr(chunks),
        chunk_results.ptr,
        N,
        void_ptr(scores),
        0.0,
        C,
        aux.ptr,
        path.ptr,
        moves.ptr,
        ffi.NULL,
        sequence.ptr,
        qstring.ptr,
        scale,
        offset,
        beam_width,
        beam_cut,
        blank_score,
    ]

    if return_hp_probs:
        # get hp scores before they are overwritten
        hp_trans_prob = scores[:, :, HP_STATES[C]].contiguous().cpu()
        hp_stay_probs = torch.full_like(hp_trans_prob, blank_score)
        hp_move_probs = torch.softmax(
            torch.stack([hp_trans_prob, hp_stay_probs], -1).float(),  # softmax not implemented for half
            dim=-1
        )[:, :, :, 0]

    lib.host_back_guide_step(*args)
    lib.host_beam_search_step(*args)
    lib.host_compute_posts_step(*args)

    if return_hp_probs:
        posts = scores.view(N, -1)[:, :(T + 1) * (C // 4)].view((N, T + 1, C // 4))
        hp_state_posts = posts[:, :T, HP_STATES[C // 4]].contiguous().cpu()

    lib.host_run_decode(*args, int(move_pad))

    moves_ = moves.data.reshape(N, -1).cpu()
    sequence_ = sequence.data.reshape(N, -1).cpu()
    qstring_ = qstring.data.reshape(N, -1).cpu()

    if return_hp_probs:
        return sequence_, qstring_, moves_, hp_move_probs, hp_state_posts
    else:
        return sequence_, qstring_, moves_
