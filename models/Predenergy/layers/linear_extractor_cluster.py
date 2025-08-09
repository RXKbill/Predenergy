import paddle
import paddle.nn as nn
from paddle.distribution import Normal
from .linear_pattern_extractor import Linear_extractor as expert
from .distributional_router_encoder import encoder
from .RevIN import RevIN
from einops import rearrange


class SparseDispatcher(object):

    def __init__(self, num_experts, gates):
        """Create a SparseDispatcher."""
        self._gates = gates
        self._num_experts = num_experts
        # sort experts
        sorted_experts, index_sorted_experts = paddle.nonzero(gates).sort(0)
        # drop indices
        _, self._expert_index = sorted_experts.split(1, axis=1)
        # get according batch index for each expert
        self._batch_index = paddle.nonzero(gates)[index_sorted_experts[:, 1], 0]
        # calculate num samples that each expert gets
        self._part_sizes = (gates > 0).sum(0).tolist()
        # expand gates to match with self._batch_index
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = paddle.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, inp):
        inp_exp = inp[self._batch_index].squeeze(1)
        return paddle.split(inp_exp, self._part_sizes, axis=0)

    def combine(self, expert_out, multiply_by_gates=True):
        # apply exp to expert outputs, so we are not longer in log space
        stitched = paddle.concat(expert_out, 0)
        if multiply_by_gates:
            # stitched = stitched.mul(self._nonzero_gates)
            stitched = paddle.einsum("i...,ij->i...", stitched, self._nonzero_gates)

        shape = list(expert_out[-1].shape)
        shape[0] = self._gates.shape[0]
        zeros = paddle.zeros(shape, dtype=stitched.dtype)
        # combine samples that have been processed by the same k experts
        combined = paddle.scatter_add(zeros, self._batch_index, stitched.astype('float32'))
        return combined

    def expert_to_gates(self):
        # split nonzero gates for each expert
        return paddle.split(self._nonzero_gates, self._part_sizes, axis=0)


class Linear_extractor_cluster(nn.Layer):

    def __init__(self, config):
        super(Linear_extractor_cluster, self).__init__()
        self.noisy_gating = config.noisy_gating
        self.num_experts = config.num_experts
        self.input_size = config.seq_len
        self.k = config.k
        # instantiate experts
        self.experts = nn.LayerList([expert(config) for _ in range(self.num_experts)])
        self.W_h = nn.Parameter(paddle.eye(self.num_experts))
        self.gate = encoder(config)
        self.noise = encoder(config)

        self.n_vars = config.enc_in
        self.revin = RevIN(self.n_vars)

        self.CI = config.CI
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.register_buffer("mean", paddle.to_tensor([0.0]))
        self.register_buffer("std", paddle.to_tensor([1.0]))
        assert self.k <= self.num_experts

    def cv_squared(self, x):
        eps = 1e-10
        # if only num_experts = 1

        if x.shape[0] == 1:
            return paddle.to_tensor([0], dtype=x.dtype)
        return x.astype('float32').var() / (x.astype('float32').mean() ** 2 + eps)

    def _gates_to_load(self, gates):
        return (gates > 0).sum(0)

    def _prob_in_top_k(
        self, clean_values, noisy_values, noise_stddev, noisy_top_values
    ):
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = (
            paddle.arange(batch, dtype=clean_values.dtype) * m + self.k
        )
        threshold_if_in = paddle.unsqueeze(
            paddle.gather(top_values_flat, 0, threshold_positions_if_in), 1
        )
        is_in = paddle.greater_than(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = paddle.unsqueeze(
            paddle.gather(top_values_flat, 0, threshold_positions_if_out), 1
        )
        # is each value currently in the top k.
        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in) / noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out) / noise_stddev)
        prob = paddle.where(is_in, prob_if_in, prob_if_out)
        return prob

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        clean_logits = self.gate(x)

        if self.noisy_gating and train:
            raw_noise_stddev = self.noise(x)
            noise_stddev = self.softplus(raw_noise_stddev) + noise_epsilon
            noise = paddle.randn_like(clean_logits)
            noisy_logits = clean_logits + (noise * noise_stddev)
            logits = noisy_logits @ self.W_h
        else:
            logits = clean_logits

        # calculate topk + 1 that will be needed for the noisy gates
        logits = self.softmax(logits)
        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)
        top_k_logits = top_logits[:, : self.k]
        top_k_indices = top_indices[:, : self.k]
        top_k_gates = top_k_logits / (
            top_k_logits.sum(1, keepdim=True) + 1e-6
        )  # normalization

        zeros = paddle.zeros_like(logits)
        gates = paddle.scatter(zeros, 1, top_k_indices, top_k_gates)

        if self.noisy_gating and self.k < self.num_experts and train:
            load = (
                self._prob_in_top_k(
                    clean_logits, noisy_logits, noise_stddev, top_logits
                )
            ).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load

    def forward(self, x, loss_coef=1):
        gates, load = self.noisy_top_k_gating(x, self.training)
        # calculate importance loss
        importance = gates.sum(0)
        loss = self.cv_squared(importance) + self.cv_squared(load)
        loss *= loss_coef

        dispatcher = SparseDispatcher(self.num_experts, gates)
        if self.CI:
            x_norm = rearrange(x, "(x y) l c -> x l (y c)", y=self.n_vars)
            x_norm = self.revin(x_norm, "norm")
            x_norm = rearrange(x_norm, "x l (y c) -> (x y) l c", y=self.n_vars)
        else:
            x_norm = self.revin(x, "norm")

        expert_inputs = dispatcher.dispatch(x_norm)

        gates = dispatcher.expert_to_gates()
        expert_outputs = [
            self.experts[i](expert_inputs[i]) for i in range(self.num_experts)
        ]
        y = dispatcher.combine(expert_outputs)

        return y, loss
