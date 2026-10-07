# spellchecker:off
# Vendored numerical kernels; see PROVENANCE.md for sources and explicit changes.
'MIT License\n\nCopyright (c) 2021-2023 TorchQuantum Authors\n\nPermission is hereby granted, free of charge, to any person obtaining a copy\nof this software and associated documentation files (the "Software"), to deal\nin the Software without restriction, including without limitation the rights\nto use, copy, modify, merge, publish, distribute, sublicense, and/or sell\ncopies of the Software, and to permit persons to whom the Software is\nfurnished to do so, subject to the following conditions:\n\nThe above copyright notice and this permission notice shall be included in all\ncopies or substantial portions of the Software.\n\nTHE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR\nIMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,\nFITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE\nAUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER\nLIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,\nOUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE\nSOFTWARE.\n'

from typing import Union
import numpy as np
import torch
from torch import nn
from .svd import truncated_svd_gesdd as svd_decomposition

C_DTYPE = torch.complex128
F_DTYPE = torch.float64


class NoisyCombTNDevice(nn.Module):
    svd_decomposition = staticmethod(svd_decomposition)

    def __init__(
        self,
        n_wires: int,
        n_dims: int,
        device: Union[torch.device, str] = "cpu",
        record_op: bool = False,
        dtype: str = "complex",
        ctol: float = 1e-09,
        otol: float = 1e-12,
        cmbd: int = 64,
        ombd: int = 16,
        noise_model: dict = None,
    ):
        """A quantum device that contains the quantum density matrix.
        Args:
            n_wires: number of qubits
            device_name: name of the quantum device
            device: which classical computing device to use, 'cpu' or 'cuda'
            record_op: whether to record the operations on the quantum device and then
                they can be used to construct a static computation graph
            dtype : str
                Type of the TN
            ctol : float
                Tolerance for SVD of the closed system
            otol : float
                Tolerance for the SVD of the open system
            cmbd : int
                Maximum bond dimension for the SVD of the closed system
            ombd : int
                Maximum bond dimension for the SVD of the closed system
        """
        super().__init__()
        self.n_wires = n_wires * n_dims
        self.n_wires_per_dim = n_wires
        self.n_dims = n_dims
        self.device_name = "noisy_comb_tn"
        self.device = device
        self.iso_center = 0
        self.dtype = C_DTYPE if dtype == "complex" else F_DTYPE
        self.osys_dim = [1] * self.n_wires
        self.ctol = ctol
        self.otol = otol
        self.cmbd = cmbd
        self.ombd = ombd
        self.states = []
        for ii in range(self.n_wires):
            _state = torch.zeros(2, dtype=self.dtype)
            _state[0] = 1
            new_shape = [1, 2, 1, 1] if ii % n_wires == 0 else [1, 2, 1]
            _state = torch.reshape(_state, new_shape).to(self.device)
            self.register_buffer(f"state_{ii}", _state)
            self.states.append(_state)
        self.record_op = record_op
        self.op_history = []
        self.iso_towards(n_wires - 1)
        if noise_model is None:
            noise_model = {}
        self.err_slope = noise_model.get("err_slope", 0.00143)
        self.err_intercept = noise_model.get("err_intercept", 0.00021)
        self.use_constant = noise_model.get("use_constant", False)
        self.zero_dim_noisy = noise_model.get("zero_dim_noisy", False)

    def __getitem__(self, idx):
        for tidx, tt in enumerate(self.states):
            if torch.isnan(tt).any():
                raise RuntimeError(f"Nans in the {tidx} tensor")
        return self.states[idx]

    def __setitem__(self, idx, val):
        self.states[idx] = val

    def overlap(self, other):
        """
        Other should be in the pure state form,
        i.e. local dimensions 2
        """
        dim_tens = []
        for dd in range(self.n_dims):
            for ii in range(self.n_wires_per_dim - 1, -1, -1):
                idx = ii + self.n_wires_per_dim * dd
                if ii == self.n_wires_per_dim - 1:
                    st = self[idx]
                    st = st.reshape(st.shape[0], 2, self.osys_dim[idx], st.shape[2])
                    tens = torch.tensordot(st, other[idx].conj(), ([1, 3], [1, 2]))
                    tens = tens.permute(0, 2, 1).reshape(-1, self.osys_dim[idx])
                    tens = torch.tensordot(tens, tens.conj(), ([1], [1]))
                elif ii > 0:
                    st = self[idx]
                    st = st.reshape(st.shape[0], 2, self.osys_dim[idx], st.shape[2])
                    st = (
                        torch
                        .tensordot(st, other[idx].conj(), ([1], [1]))
                        .permute(0, 3, 1, 2, 4)
                        .reshape(
                            self[idx].shape[0] * other[idx].shape[0],
                            self.osys_dim[idx],
                            self[idx].shape[2] * other[idx].shape[2],
                        )
                    )
                    stc = st.conj()
                    tens = torch.tensordot(st, tens, ([2], [0]))
                    tens = torch.tensordot(tens, stc, ([1, 2], [1, 2]))
                else:
                    st = self[idx]
                    st = st.reshape(st.shape[0], 2, self.osys_dim[idx], *st.shape[2:])
                    st = (
                        torch
                        .tensordot(st, other[idx].conj(), ([1], [1]))
                        .permute(0, 4, 1, 2, 5, 3, 6)
                        .reshape(
                            self[idx].shape[0] * other[idx].shape[0],
                            self.osys_dim[idx],
                            self[idx].shape[2] * other[idx].shape[2],
                            self[idx].shape[3] * other[idx].shape[3],
                        )
                    )
                    stc = st.conj()
                    tens = torch.tensordot(st, tens, ([2], [0])).permute(0, 1, 3, 2)
                    tens = torch.tensordot(tens, stc, ([1, 2], [1, 2])).permute(0, 2, 1, 3)
                    dim_tens.append(tens)
        tens = dim_tens[0].reshape(dim_tens[0].shape[2:])
        for tt in dim_tens[1:]:
            tens = torch.tensordot(tens, tt, ([0, 1], [0, 1]))
        return tens

    def apply_one_site_operator(self, wires, matrix):
        state = self[wires]
        sshape = state.shape
        matrix = matrix.to(dtype=self.dtype)
        state = state.reshape(sshape[0], 2, self.osys_dim[wires], *sshape[2:])
        state = torch.tensordot(state, matrix, ([1], [1]))
        if len(sshape) == 3:
            state = torch.permute(state, [0, 3, 1, 2])
        elif len(sshape) == 4:
            state = torch.permute(state, [0, 4, 1, 2, 3])
        self[wires] = state.reshape(sshape[0], 2 * self.osys_dim[wires], *sshape[2:])

    def apply_two_sites_operator(self, idx, jdx, matrix, dirc="R"):
        matrix = matrix.reshape(2, 2, 2, 2).to(dtype=self.dtype)
        if idx > jdx:
            matrix = torch.permute(matrix, [1, 0, 3, 2])
        minid = min(idx, jdx)
        maxid = max(idx, jdx)
        to_iso = minid if np.abs(self.iso_center - minid) < np.abs(self.iso_center - maxid) else maxid
        self.iso_towards(to_iso)
        mint = self[minid]
        maxt = self[maxid]
        if idx % self.n_wires_per_dim == 0 and jdx % self.n_wires_per_dim == 0:
            two_tens = torch.tensordot(mint, maxt, ([3], [0]))
            two_tens = two_tens.reshape(
                mint.shape[0],
                2,
                self.osys_dim[minid],
                mint.shape[-2],
                2,
                self.osys_dim[maxid],
                maxt.shape[2],
                maxt.shape[3],
            )
            two_tens = (
                torch
                .tensordot(two_tens, matrix, ([1, 4], [2, 3]))
                .permute(0, 6, 1, 2, 7, 3, 4, 5)
                .reshape(-1, np.prod(maxt.shape[1:]))
            )
            uu, ss, vv = self.svd_decomposition(two_tens, self.cmbd, rel_tol=self.ctol)
            vv = vv.T.conj()
            ss = ss.to(vv.dtype)
            if dirc == "R":
                rr = torch.matmul(torch.diag(ss), vv)
            else:
                uu = torch.matmul(uu, torch.diag(ss))
                rr = vv
            chi = uu.shape[1]
            mint = uu.reshape(*mint.shape[:3], chi)
            maxt = rr.reshape((chi, *maxt.shape[1:]))
        elif minid % self.n_wires_per_dim == 0:
            two_tens = torch.tensordot(mint, maxt, ([2], [0]))
            two_tens = two_tens.reshape(
                mint.shape[0], 2, self.osys_dim[minid], mint.shape[-1], 2, self.osys_dim[maxid], maxt.shape[2]
            )
            two_tens = (
                torch
                .tensordot(two_tens, matrix, ([1, 4], [2, 3]))
                .permute(0, 5, 1, 2, 6, 3, 4)
                .reshape(-1, np.prod(maxt.shape[1:]))
            )
            uu, ss, vv = self.svd_decomposition(two_tens, self.cmbd, rel_tol=self.ctol)
            vv = vv.T.conj()
            ss = ss.to(vv.dtype)
            if dirc == "R":
                rr = torch.matmul(torch.diag(ss), vv)
            else:
                uu = torch.matmul(uu, torch.diag(ss))
                rr = vv
            chi = uu.shape[1]
            mint = uu.reshape(*mint.shape[:2], mint.shape[3], chi)
            mint = torch.permute(mint, (0, 1, 3, 2))
            maxt = rr.reshape((chi, *maxt.shape[1:]))
        else:
            two_tens = torch.tensordot(mint, maxt, ([2], [0]))
            two_tens = two_tens.reshape(mint.shape[0], 2, self.osys_dim[minid], 2, self.osys_dim[maxid], maxt.shape[2])
            two_tens = (
                torch
                .tensordot(two_tens, matrix, ([1, 3], [2, 3]))
                .permute(0, 4, 1, 5, 2, 3)
                .reshape(np.prod(mint.shape[:2]), -1)
            )
            uu, ss, vv = self.svd_decomposition(two_tens, self.cmbd, rel_tol=self.ctol)
            vv = vv.T.conj()
            ss = ss.to(vv.dtype)
            if dirc == "R":
                rr = torch.matmul(torch.diag(ss), vv)
            else:
                uu = torch.matmul(uu, torch.diag(ss))
                rr = vv
            chi = uu.shape[1]
            mint = uu.reshape((*mint.shape[:2], chi))
            maxt = rr.reshape((chi, *maxt.shape[1:]))
        self[minid] = mint
        self[maxid] = maxt
        if dirc == "R":
            self.iso_center = maxid
        else:
            self.iso_center = minid

    def iso_towards(self, jdx):
        nwd = self.n_wires_per_dim
        if jdx % nwd == 0 and self.iso_center % nwd == 0:
            step = 1 if jdx > self.iso_center else -1
            idxs = [nwd * ii for ii in range(self.iso_center // nwd, jdx // nwd + step, step)]
        elif jdx // nwd == self.iso_center // nwd:
            step = 1 if jdx > self.iso_center else -1
            idxs = [ii for ii in range(self.iso_center, jdx + step, step)]
        else:
            idxs = [ii for ii in range(self.iso_center, self.iso_center // nwd * nwd - 1, -1)]
            new_iso = idxs[-1]
            step = 1 if jdx > new_iso else -1
            idxs += [nwd * ii for ii in range(new_iso // nwd, jdx // nwd + step, step)]
            new_iso = idxs[-1]
            idxs += [ii for ii in range(new_iso, jdx + 1)]
        for idx in idxs:
            self.move_iso_one_step(idx)

    def move_iso_one_step(self, jdx):
        idx = self.iso_center
        if jdx == idx:
            return
        it = self[idx]
        jt = self[jdx]
        if idx % self.n_wires_per_dim == 0 and jdx % self.n_wires_per_dim != 0:
            tt = it.permute(0, 1, 3, 2)
            qq, rr = torch.linalg.qr(tt.reshape(-1, tt.shape[-1]))
            self[idx] = qq.reshape(*tt.shape[:-1], -1).permute(0, 1, 3, 2)
            self[jdx] = torch.tensordot(rr, jt, ([1], [0]))
        elif jdx % self.n_wires_per_dim == 0 and idx % self.n_wires_per_dim != 0:
            if idx < jdx:
                qq, rr = torch.linalg.qr(it.reshape(-1, it.shape[-1]))
                self[idx] = qq.reshape(*it.shape[:-1], -1)
                self[jdx] = torch.tensordot(rr, jt, ([1], [0]))
            else:
                qq, rr = torch.linalg.qr(it.reshape(it.shape[0], -1).T)
                self[idx] = qq.T.reshape(-1, *it.shape[1:])
                self[jdx] = torch.tensordot(jt, rr.T, ([2], [1])).permute(0, 1, 3, 2)
        elif idx < jdx:
            qq, rr = torch.linalg.qr(it.reshape(-1, it.shape[-1]))
            self[idx] = qq.reshape(*it.shape[:-1], -1)
            self[jdx] = torch.tensordot(rr, jt, ([1], [0]))
        else:
            qq, rr = torch.linalg.qr(it.reshape(it.shape[0], -1).T)
            self[idx] = qq.T.reshape(-1, *it.shape[1:])
            self[jdx] = torch.tensordot(jt, rr.T, ([-1], [1]))
        self.iso_center = jdx


# spellchecker:on
