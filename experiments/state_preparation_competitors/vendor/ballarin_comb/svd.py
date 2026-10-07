# spellchecker:off
# Vendored numerical kernels; see PROVENANCE.md for sources and explicit changes.
'MIT License\n\nCopyright (c) 2021-2023 TorchQuantum Authors\n\nPermission is hereby granted, free of charge, to any person obtaining a copy\nof this software and associated documentation files (the "Software"), to deal\nin the Software without restriction, including without limitation the rights\nto use, copy, modify, merge, publish, distribute, sublicense, and/or sell\ncopies of the Software, and to permit persons to whom the Software is\nfurnished to do so, subject to the following conditions:\n\nThe above copyright notice and this permission notice shall be included in all\ncopies or substantial portions of the Software.\n\nTHE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR\nIMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,\nFITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE\nAUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER\nLIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,\nOUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE\nSOFTWARE.\n'

import torch


def safe_inverse(x, epsilon=1e-12):
    return x / (x**2 + epsilon)


def safe_inverse_2(x, epsilon):
    x[abs(x) < epsilon] = float("inf")
    return x.pow(-1)


class SVDGESDD(torch.autograd.Function):
    @staticmethod
    def forward(self, A, cutoff, diagnostics):
        """
        :param A: rank-2 tensor
        :type A: torch.Tensor
        :param cutoff: cutoff for backward function
        :type cutoff: torch.Tensor
        :param diagnostics: optional dictionary for debugging purposes
        :type diagnostics: dict
        :return: U, S, V
        :rtype: torch.Tensor, torch.Tensor, torch.Tensor

        Computes SVD decompostion of matrix :math:`A = USV^\\dagger`.
        """
        U, S, Vh = torch.linalg.svd(A, full_matrices=False)
        V = Vh.transpose(-2, -1).conj()
        self.save_for_backward(U, S, V, cutoff)
        self.diagnostics = diagnostics
        return (U, S, V)

    @staticmethod
    def backward(self, gu, gsigma, gv):
        """
        :param gu: gradient on U
        :type gu: torch.Tensor
        :param gsigma: gradient on S
        :type gsigma: torch.Tensor
        :param gv: gradient on V
        :type gv: torch.Tensor
        :return: gradient
        :rtype: torch.Tensor

        Computes backward gradient for SVD, adopted from
        https://github.com/pytorch/pytorch/blob/v1.10.2/torch/csrc/autograd/FunctionsManual.cpp

        For complex-valued input there is an additional term, see

            * https://giggleliu.github.io/2019/04/02/einsumbp.html
            * https://arxiv.org/abs/1909.02659

        The backward is regularized following

            * https://github.com/wangleiphy/tensorgrad/blob/master/tensornets/adlib/svd.py
            * https://arxiv.org/abs/1903.09650

        using

        .. math::
            S_i/(S^2_i-S^2_j) = (F_{ij}+G_{ij})/2\\ \\ \\textrm{and}\\ \\ S_j/(S^2_i-S^2_j) = (F_{ij}-G_{ij})/2

        where

        .. math::
            F_{ij}=1/(S_i-S_j),\\ G_{ij}=1/(S_i+S_j)
        """
        diagnostics = self.diagnostics
        u, sigma, v, eps = self.saved_tensors
        m = u.size(0)
        n = v.size(0)
        k = sigma.size(0)
        sigma_scale = sigma[0]
        if u.size(-2) != u.size(-1) or v.size(-2) != v.size(-1):
            u = u.narrow(-1, 0, k)
            v = v.narrow(-1, 0, k)
            if not gu is None:
                gu = gu.narrow(-1, 0, k)
            if not gv is None:
                gv = gv.narrow(-1, 0, k)
        vh = v.conj().transpose(-2, -1)
        if not gsigma is None:
            sigma_term = u * gsigma.unsqueeze(-2) @ vh
        else:
            sigma_term = torch.zeros(m, n, dtype=u.dtype, device=u.device)
        if gv is None and gv is None:
            return (sigma_term, None, None)
        sigma_inv = safe_inverse_2(sigma.clone(), sigma_scale * eps)
        F = sigma.unsqueeze(-2) - sigma.unsqueeze(-1)
        F = safe_inverse(F, sigma_scale * eps)
        F.diagonal(0, -2, -1).fill_(0)
        G = sigma.unsqueeze(-2) + sigma.unsqueeze(-1)
        G = safe_inverse(G, sigma_scale * eps)
        G.diagonal(0, -2, -1).fill_(0)
        uh = u.conj().transpose(-2, -1)
        if not gu is None:
            guh = gu.conj().transpose(-2, -1)
            u_term = u @ (F + G).mul(uh @ gu - guh @ u) * 0.5
            if m > k:
                # Apply (I - U U^H) to the skinny factor without an m-by-m projector.
                weighted = gu * sigma_inv.unsqueeze(-2)
                u_term = u_term + weighted - u @ (uh @ weighted)
            u_term = u_term @ vh
        else:
            u_term = torch.zeros(m, n, dtype=u.dtype, device=u.device)
        # Preserve the original addition order while releasing full-matrix terms.
        dA = u_term
        dA.add_(sigma_term)
        del sigma_term, u_term
        if not gv is None:
            gvh = gv.conj().transpose(-2, -1)
            v_term = (F - G).mul(vh @ gv - gvh @ v) @ vh * 0.5
            if n > k:
                # Apply the right projector without an n-by-n intermediate.
                weighted = sigma_inv.unsqueeze(-1) * gvh
                v_term = v_term + weighted - (weighted @ v) @ vh
            v_term = u @ v_term
        else:
            v_term = torch.zeros(m, n, dtype=u.dtype, device=u.device)
        dA.add_(v_term)
        del v_term
        if u.is_complex() or v.is_complex():
            L = (uh @ gu).diagonal(0, -2, -1)
            L.real.zero_()
            L.imag.mul_(sigma_inv)
            imag_term = u * L.unsqueeze(-2) @ vh
            dA.add_(imag_term)
            del imag_term
        if not diagnostics is None:
            print(f"{diagnostics} {dA.abs().max()} {sigma.max()}")
        return (dA, None, None)


def _keep_multiplets(U, S, V, chi, eps_multiplet, abs_tol):
    chi_new = chi
    gaps = S[: chi + 1].clone().detach()
    gaps[gaps < abs_tol] = 0.0
    gaps = (gaps[:chi] - S[1 : chi + 1]) / (gaps[:chi] + 1e-16)
    gaps[gaps > 1.0] = 0.0
    if gaps[chi - 1] < eps_multiplet:
        for i in range(chi - 1, -1, -1):
            if gaps[i] > eps_multiplet:
                chi_new = i
                break
    St = S[:chi].clone()
    St[chi_new + 1 :] = 0.0
    Ut = U[:, : St.shape[0]].clone()
    Ut[:, chi_new + 1 :] = 0.0
    Vt = V[:, : St.shape[0]].clone()
    Vt[:, chi_new + 1 :] = 0.0
    return (Ut, St, Vt)


def truncated_svd_gesdd(
    M,
    chi,
    abs_tol=1e-14,
    rel_tol=None,
    ad_decomp_reg=1e-12,
    keep_multiplets=False,
    eps_multiplet=1e-12,
    verbosity=0,
    diagnostics=None,
):
    """
    :param M: matrix of dimensions :math:`N \\times L`
    :param chi: desired maximal rank :math:`\\chi`
    :param abs_tol: absolute tolerance on minimal singular value
    :param rel_tol: relative tolerance on minimal singular value
    :param keep_multiplets: truncate spectrum down to last complete multiplet
    :param eps_multiplet: allowed splitting within multiplet
    :param verbosity: logging verbosity
    :type M: torch.tensor
    :type chi: int
    :type abs_tol: float
    :type rel_tol: float
    :type keep_multiplets: bool
    :type eps_multiplet: float
    :type verbosity: int
    :return: leading :math:`\\chi` left singular vectors U, right singular vectors V, and
             singular values S
    :rtype: torch.tensor, torch.tensor, torch.tensor

    Returns leading :math:`\\chi`-singular triples of a matrix M by computing the full
    SVD :math:`M= USV^T`. Returned tensors have dimensions

    .. math:: dim(U)=(N,\\chi),\\ dim(S)=(\\chi,\\chi),\\ \\textrm{and}\\ dim(V)=(L,\\chi)
    """
    reg = torch.as_tensor(ad_decomp_reg, dtype=M.real.dtype if M.is_complex() else M.dtype, device=M.device)
    U, S, V = SVDGESDD.apply(M, reg, diagnostics)
    chi_new = chi
    if keep_multiplets and chi < S.shape[0]:
        gaps = S[: chi + 1].clone().detach()
        gaps[gaps < abs_tol] = 0.0
        gaps = (gaps[:chi] - S[1 : chi + 1]) / (gaps[:chi] + 1e-16)
        gaps[gaps > 1.0] = 0.0
        if gaps[chi - 1] < eps_multiplet:
            for i in range(chi - 1, -1, -1):
                if gaps[i] > eps_multiplet:
                    chi_new = i
                    break
        St = S[:chi].clone()
        St[chi_new + 1 :] = 0.0
        Ut = U[:, : St.shape[0]].clone()
        Ut[:, chi_new + 1 :] = 0.0
        Vt = V[:, : St.shape[0]].clone()
        Vt[:, chi_new + 1 :] = 0.0
        return (Ut, St, Vt)
    norm = (torch.flip(S, (0,)) ** 2).cumsum(0) / (S**2).sum(0)
    cut = torch.nonzero(norm > rel_tol)[0]
    chi_now = len(S)
    chi_by_norm = len(S) - cut[0] if len(cut) > 0 else chi_now
    rank = min(chi_now, min(chi, chi_by_norm))
    singvals_kept = S[:rank]
    singvals_cutted = S[rank:]
    norm_kept = (singvals_kept**2).sum()
    norm_trunc = (singvals_cutted**2).sum()
    normalization_factor = torch.sqrt(norm_kept) / torch.sqrt(norm_kept + norm_trunc)
    singvals_kept = singvals_kept / normalization_factor
    St = singvals_kept
    Ut = U[:, : St.shape[0]]
    Vt = V[:, : St.shape[0]]
    if verbosity:
        return (Ut, St, Vt, norm_trunc)
    return (Ut, St, Vt)


# spellchecker:on
