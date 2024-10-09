import torch
from koi._runtime import lib, ffi
from collections import namedtuple
from koi.utils import void_ptr

semiring = namedtuple('semiring', ('zero', 'one', 'mul', 'sum', 'dsum'))

def max_grad(x, dim=0):
    return torch.zeros_like(x).scatter_(dim, x.argmax(dim, True), 1.0)

Log = semiring(zero=-1e38, one=0., mul=torch.add, sum=torch.logsumexp, dsum=torch.softmax)
Max = semiring(zero=-1e38, one=0., mul=torch.add, sum=(lambda x, dim=0: torch.max(x, dim=dim)[0]), dsum=max_grad)

def grad(f, x):
    x = x.detach().requires_grad_()
    with torch.enable_grad():
        y = f(x)
        return torch.autograd.grad(y, x)[0].detach()

class SequenceDist():
    def __init__(self):
        pass

    def logZ(self, scores, S:semiring=Log):
        raise NotImplementedError

    def viterbi(self, scores):
        raise NotImplementedError

    def ctc_loss(self, scores, targets, target_lengths):
        raise NotImplementedError

    def posteriors(self, scores, S:semiring=Log):
        f = lambda x: self.logZ(x, S).sum()
        return grad(f, scores)


class LogZ(torch.autograd.Function):
    @staticmethod
    def forward(ctx, stay_scores, move_scores, target_lengths, fwd_bwd_impl, S:semiring):
        T, N, L = stay_scores.shape

        alpha = stay_scores.new_full((T + 1, N, L), S.zero)
        alpha[0, :, 0] = S.one

        beta_stay = stay_scores.new_full((T, N, L), S.zero)
        beta_move = stay_scores.new_full((T, N, L), S.zero)
        beta_T = stay_scores.new_full((N, L), S.zero)
        beta_T[torch.arange(N), target_lengths - 1] = S.one

        fwd_bwd_impl(alpha, beta_T, beta_stay, beta_move, stay_scores, move_scores, S)

        g = S.dsum(torch.cat([S.mul(alpha[:-1], beta_stay), S.mul(alpha[:-1], beta_move)], dim=2), dim=2)

        ctx.save_for_backward(g.reshape(T, N, 2, L))
        return dot(alpha[-1], beta_T, S)

    @staticmethod
    def backward(ctx, grad):
        g = ctx.saved_tensors[0] * grad[None, :, None, None]
        return g[:, :, 0], g[:, :, 1, :-1], None, None, None


def dot(x, y, S=Log, dim=-1):
    return S.sum(S.mul(x, y), dim=dim)


def _simple_lattice_fwd_bwd_cu_loop(alpha, beta_T, beta_stay, beta_move, stay_scores, move_scores, S:semiring):
    if stay_scores.dtype != torch.float32:
        raise NotImplementedError("Only fp32 supported")
    if S != Log:
        raise NotImplementedError("Only Log Semiring supported")

    T, N, L = stay_scores.shape
    beta = alpha.new_full(alpha.shape, S.zero)
    beta[-1] = beta_T

    with torch.cuda.device(torch.device('cuda', stay_scores.device.index)):
        lib.fwd_bwd_logspace_loop_host(T,
                                       N,
                                       L,
                                       void_ptr(alpha),
                                       void_ptr(beta),
                                       void_ptr(beta_stay),
                                       void_ptr(beta_move),
                                       void_ptr(stay_scores),
                                       void_ptr(move_scores),
        )


def _simple_lattice_fwd_bwd_cu(alpha, beta_T, beta_stay, beta_move, stay_scores, move_scores, S:semiring):
    if stay_scores.dtype != torch.float32:
        raise NotImplementedError("Only fp32 supported")
    if S != Log:
        raise NotImplementedError("Only Log Semiring supported")

    T, N, L = stay_scores.shape
    if L > 1024: #exceeds max threads per block
        return _simple_lattice_fwd_bwd_cu_loop(alpha, beta_T, beta_stay, beta_move, stay_scores, move_scores, S)

    with torch.cuda.device(torch.device('cuda', stay_scores.device.index)):
        lib.fwd_bwd_logspace_host(T,
                                  N,
                                  L,
                                  void_ptr(alpha),
                                  void_ptr(beta_T),
                                  void_ptr(beta_stay),
                                  void_ptr(beta_move),
                                  void_ptr(stay_scores),
                                  void_ptr(move_scores),
        )


def logZ_cu(stay_scores, move_scores, target_lengths, S:semiring=Log):
    return LogZ.apply(stay_scores, move_scores, target_lengths, _simple_lattice_fwd_bwd_cu, S)


def viterbi_alignments(stay_scores, move_scores, target_lengths):
    target_lengths = target_lengths.to(stay_scores.device)
    stay_scores, move_scores = stay_scores.detach().requires_grad_(), move_scores.detach().requires_grad_()
    logZ_cu(stay_scores, move_scores, target_lengths, Max).sum().backward()
    alignments = stay_scores.grad.clone()
    alignments[:, :, :-1] += move_scores.grad
    return alignments


def fwd_scores_cu_sparse(Ms, idx, v0, S:semiring=Log, K=1):
    if Ms.dtype != torch.float32:
        raise NotImplementedError("Only fp32 supported")
    if K != 1:
        raise NotImplementedError("Only K=1 supported")

    T, N, C, NZ = Ms.shape
    alphas = Ms.new_full((T+1, N, C), S.zero)
    idx = idx.to(dtype=torch.int, device=Ms.device)

    if S == Log:
        if NZ not in [3, 5]:
            raise NotImplementedError("Only NZ=3 or NZ=5 supported")
        host_fn = getattr(lib, f"fwd_scores_host_sparse_log_NZ{NZ}")
        with torch.cuda.device(Ms.device.index):
            host_fn(void_ptr(alphas),
                    void_ptr(Ms),
                    void_ptr(v0),
                    void_ptr(idx),
                    T,
                    N,
                    C)
    elif S == Max:
        if NZ not in [5]:
            raise NotImplementedError("Only NZ=5 supported")
        with torch.cuda.device(Ms.device.index):
            lib.fwd_scores_host_sparse_max(void_ptr(alphas),
                                           void_ptr(Ms),
                                           void_ptr(v0),
                                           void_ptr(idx),
                                           T,
                                           N,
                                           C)
    else:
        raise NotImplementedError("Semiring not supported")

    return alphas


def bwd_scores_cu_sparse(Ms, idx, vT, S:semiring=Log, K=1):
    if Ms.dtype != torch.float32:
        raise NotImplementedError("Only fp32 supported")
    if K != 1:
        raise NotImplementedError("Only K=1 supported")

    T, N, C, NZ = Ms.shape
    betas = Ms.new_full((T+1, N, C), S.zero)
    idx_T = idx.flatten().argsort().to(dtype=torch.int, device=Ms.device) #transpose

    if S == Log:
        if NZ not in [3, 5]:
            raise NotImplementedError("Only NZ=3 or NZ=5 supported")
        host_fn = getattr(lib, f"bwd_scores_host_sparse_log_NZ{NZ}")
        with torch.cuda.device(Ms.device.index):
            host_fn(void_ptr(betas),
                    void_ptr(Ms),
                    void_ptr(vT),
                    void_ptr(idx_T),
                    T,
                    N,
                    C)
    elif S == Max:
        if NZ not in [5]:
            raise NotImplementedError("Only NZ=5 supported")
        with torch.cuda.device(Ms.device.index):
            lib.bwd_scores_host_sparse_max(void_ptr(betas),
                                           void_ptr(Ms),
                                           void_ptr(vT),
                                           void_ptr(idx_T),
                                           T,
                                           N,
                                           C)
    else:
        raise NotImplementedError("Semiring not supported")

    return betas


def logZ_fwd_cu(Ms, idx, v0, vT, S:semiring=Log, K=4):
    if Ms.dtype != torch.float32:
        raise NotImplementedError("Only fp32 supported")
    if K != 1:
        raise NotImplementedError("Only K=1 supported")

    assert Ms.device.index is not None
    T, N, C, NZ = Ms.shape
    assert idx.shape == (C, NZ)
    idx = idx.to(dtype=torch.int, device=Ms.device)
    #idx = idx.to(dtype=torch.int)
    Ms_grad = Ms.new_full((T, N, C, NZ), S.zero)
    logZ = Ms.new_full((N, C), S.zero)

    if S == Log:
        if NZ not in [3, 5]:
            raise NotImplementedError("Only NZ=3 or NZ=5 supported")
        host_fn = getattr(lib, f"logZ_fwd_host_log_NZ{NZ}")
        with torch.cuda.device(Ms.device.index):
            host_fn(T,
                    N,
                    C,
                    K,
                    void_ptr(logZ),
                    void_ptr(Ms_grad),
                    void_ptr(Ms),
                    void_ptr(v0),
                    void_ptr(vT),
                    void_ptr(idx))
    elif S == Max:
        if NZ not in [5]:
            raise NotImplementedError("Only NZ=5 supported")
        with torch.cuda.device(Ms.device.index):
            lib.logZ_fwd_host_max(T,
                                  N,
                                  C,
                                  K,
                                  void_ptr(logZ),
                                  void_ptr(Ms_grad),
                                  void_ptr(Ms),
                                  void_ptr(v0),
                                  void_ptr(vT),
                                  void_ptr(idx))

    else:
        raise NotImplementedError("Semiring not supported")

    return S.sum(logZ, dim=1), Ms_grad


class _LogZ(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Ms, idx, v0, vT, S:semiring, K):
        idx = idx.to(device=Ms.device)
        logZ, Ms_grad = logZ_fwd_cu(Ms, idx, v0, vT, S, K)
        ctx.save_for_backward(Ms_grad, Ms, idx, vT)
        ctx.semiring = S
        ctx.K = K
        return logZ

    @staticmethod
    def backward(ctx, grad):
        Ms_grad, Ms, idx, vT = ctx.saved_tensors
        S, K = ctx.semiring, ctx.K
        T, N, C, NZ = Ms.shape
        betas = bwd_scores_cu_sparse(Ms, idx, vT, S, K=K)
        Ms_grad = S.mul(Ms_grad, betas[1:,:,:,None])
        Ms_grad = S.dsum(Ms_grad.reshape(T, N, -1), dim=2).reshape(T, N, C, NZ)
        return grad[None, :, None, None] * Ms_grad, None, None, None, None, None


def logZ_cu_sparse(Ms, idx, v0, vT, S:semiring=Log, K=1):
    return _LogZ.apply(Ms, idx, v0, vT, S, K)
