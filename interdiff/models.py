import math
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

from .modules import (
    SerialisableModule,
    VectorQuantizer,
    TransformerConfig,
    LatentActionModelConfig,
    VQConfig,
    TransformerEncoder,
)
from interdiff.utils.eval_utils import sample_from_logits

class GPT(SerialisableModule):
    """Generative Pre-trained Transformer for molecular sequence generation.
    
    Standard autoregressive language model with token embeddings, positional
    embeddings, and a language modeling head.
    
    Args:
        vocab_size: Size of the vocabulary.
        n_layer: Number of transformer layers.
        n_head: Number of attention heads.
        n_embd: Embedding dimension.
        dropout: Dropout probability.
        bias: Whether to use bias in linear layers.
        context_length: Maximum sequence length.
        lm_head_out_size: Output size of the language model head.
        pad_token_id: Token ID for padding.
        bos_token_id: Token ID for beginning of sequence.
        eos_token_id: Token ID for end of sequence.
    """

    def __init__(self, 
                 vocab_size: int = 2048,
                 n_layer: int = 6,
                 n_head: int = 6,
                 n_embd: int = 384,
                 dropout: float = 0.2,
                 bias: bool = False,
                 context_length: int = 128,
                 lm_head_out_size: int = 2048,
                 pad_token_id: int = 0,
                 bos_token_id: int = 3,
                 eos_token_id: int = 4):
        
        super().__init__()

        config = TransformerConfig(
            vocab_size=vocab_size,
            n_layer=n_layer,
            n_head=n_head,
            n_embd=n_embd,
            dropout=dropout,
            bias=bias,
            context_length=context_length,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
        )
        for k, v in config.__dict__.items():
            setattr(self, k, v)
        
        self.lm_head_out_size = lm_head_out_size
        self.config = config
        self.encoder = TransformerEncoder(self.config)
        self.lm_head = nn.Linear(config.n_embd, lm_head_out_size, bias=False)
        # self.encoder.wte.weight = (
        #     self.lm_head.weight
        # )  # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer)
                )

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def forward(
        self, idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the model.
        
        Args:
            idx: Token indices of shape (batch_size, seq_len).
            
        Returns:
            Tuple containing:
                - logits: Token predictions of shape (batch_size, seq_len, vocab_size).
                - z: Hidden states of shape (batch_size, seq_len, n_embd).
        """
        z = self.encoder(idx)  # (b, t, n_embd)
        logits = self.lm_head(z)

        return logits, z

    @torch.no_grad()
    def generate(
        self,
        context: torch.Tensor | None = None,      # (B,T) or (T,)
        n_mols: int = 256,
        max_new_tokens: int = 128,                # total final length cap, not "to add"
        temperature: float = 1.0,
        top_k: int | None = None,
    ):
        """
        Batched autoregressive sampling.

        Args:
            context: LongTensor [(B, T0) or (T0,)] of token indices.
                    If None, starts from <BOS> for B = n_mols.
            n_mols: Batch size to use when context is None.
            max_new_tokens: Generate until sequence length reaches this many tokens
                            OR until all sequences emit <EOS>.
            temperature: >0 samples; <=0 switches to greedy.
            top_k: if set, restrict sampling to top_k tokens per step.

        Returns:
            LongTensor of shape (B, T_out).
        """
        device = self.lm_head.weight.device

        # --- Prepare initial context (B, T) ---
        if context is None:
            B = n_mols
            context = torch.full((B, 1), self.bos_token_id, dtype=torch.long, device=device)
        else:
            if context.dim() == 1:
                context = context.unsqueeze(0)  # (1, T)
            context = context.to(device=device, dtype=torch.long)

        # We will grow until either EOS everywhere or length hits max_new_tokens
        T0 = context.size(1)
        target_len = min(max_new_tokens, self.config.context_length)  # final length cap
        # If caller meant "add up to max_new_tokens more", replace the line above with:
        # target_len = min(T0 + max_new_tokens, self.config.context_length)

        # Track which rows are finished
        finished = torch.zeros(context.size(0), dtype=torch.bool, device=device)

        # Keep generating while there is room and not all finished
        while context.size(1) < target_len and not torch.all(finished):
            # Crop to context_length if needed
            idx_cond = context if context.size(1) <= self.config.context_length else context[:, -self.config.context_length:]

            # Forward pass -> logits for all positions
            logits, _, = self(idx_cond)                     # (B, T_ctx, vocab)
            logits = logits[:, -1, :]                         # (B, vocab) last step

            idx_next = sample_from_logits(tensor_logits=logits, temperature=temperature, top_k=top_k)  # (B,1)

            # Force already-finished rows to keep emitting EOS (no changes after EOS)
            eos = torch.full_like(idx_next, self.eos_token_id)
            idx_next = torch.where(finished.unsqueeze(1), eos, idx_next)           # (B,1)

            # Update finished mask for rows that just produced EOS
            finished = finished | (idx_next.squeeze(1) == self.eos_token_id)

            # Append
            context = torch.cat([context, idx_next], dim=1)

        return context

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """
        Estimate Model Flops Utilization (MFU) in units of A100 bfloat16 peak FLOPS
        """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.context_length
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0 / dt)  # per second
        flops_promised = 312e12  # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    def get_num_params(self, non_embedding=True):
        """Return the number of parameters in the model.
        
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        
        Args:
            non_embedding: Whether to exclude position embedding parameters.
            
        Returns:
            Number of parameters in the model.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.encoder.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        """Initialize weights for linear and embedding layers.
        
        Args:
            module: PyTorch module to initialize.
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


class LatentActionModel(SerialisableModule):
    """Latent Action Model for learning discrete action representations.
    
    Learns to encode molecular sequences into discrete latent actions using
    vector quantization, and decode sequences conditioned on these actions.
    
    Args:
        vocab_size: Size of the vocabulary.
        n_layer: Number of transformer layers.
        n_head: Number of attention heads.
        n_embd: Embedding dimension.
        dropout: Dropout probability.
        bias: Whether to use bias in linear layers.
        context_length: Maximum sequence length.
        lm_head_out_size: Output size of the language model head.
        pad_token_id: Token ID for padding.
        bos_token_id: Token ID for beginning of sequence.
        eos_token_id: Token ID for end of sequence.
        latent_action_dim: Dimension of the latent action space.
        num_latents: Number of discrete latent codes.
        entropy_weight: Weight for entropy regularization.
        vq_beta: Beta parameter for VQ commitment loss.
    """
    def __init__(self, 
                 vocab_size: int = 2048,
                 n_layer: int = 6,
                 n_head: int = 6,
                 n_embd: int = 384,
                 dropout: float = 0.2,
                 bias: bool = False,
                 context_length: int = 128,
                 lm_head_out_size: int = 2048,
                 pad_token_id: int = 0,
                 bos_token_id: int = 3,
                 eos_token_id: int = 4,
                 latent_action_dim: int = 64,
                 num_latents: int = 512,
                 entropy_weight: float = 0.01,
                 vq_beta: float = 0.25):

        # encoder is future dependent for learning and past dependent for finetuning
        # after policy distillation

        # Vector quantizer
        super().__init__()

        t_conf = TransformerConfig(
            vocab_size=vocab_size,
            n_layer=n_layer,
            n_head=n_head,
            n_embd=n_embd,
            dropout=dropout,
            bias=bias,
            context_length=context_length,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
        )

        vq_conf = VQConfig(latent_action_dim=latent_action_dim,
                           num_latents=num_latents,
                           entropy_weight=entropy_weight,
                           dropout=dropout,
                           vq_beta=vq_beta)

        kwargs_gpt = t_conf.__dict__
        kwargs_gpt['lm_head_out_size'] = lm_head_out_size

        kwargs_vq = vq_conf.__dict__

        for k, v in t_conf.__dict__.items():
            setattr(self, k, v)
        for k, v in vq_conf.__dict__.items():
            setattr(self, k, v)
        
        self.vq = VectorQuantizer(**kwargs_vq)

        # Encoder and Decoder
        self.encoder = GPT(**kwargs_gpt)
        self.decoder = GPT(**kwargs_gpt)

        self.codebook = self.vq.codebook

    def vq_encode(self, tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode tokens into quantized latent actions.
        
        Args:
            tokens: Token indices of shape (batch_size, seq_len).
            
        Returns:
            Tuple containing:
                - z_q: Quantized actions of shape (batch_size, seq_len-1, latent_action_dim).
                - vq_loss: Vector quantization loss.
                - indices: Codebook indices of shape (batch_size, seq_len-1).
        """
        B, T = tokens.shape

        _, z = self.encoder(tokens)  # (B, T, model_dim)
        # Get latent action for all future frames
        z = z[:, 1:, :]  # (B, T-1, model_dim) -> the action at time t encodes info about token at time t+1
        # Quantize only the latent action tokens
        z_flat = z.reshape(B * (T - 1), -1)  # (B*(T-1), model_dim)
        z_q, vq_loss, indices = self.vq(z_flat)
        z_q = z_q.reshape(B, T - 1, self.latent_action_dim)
        indices = indices.reshape(B, T - 1)

        return z_q, vq_loss, indices

    def decode(self, tokens: torch.Tensor, actions: torch.Tensor):
        """Decode tokens conditioned on latent actions.
        
        Args:
            tokens: Token indices of shape (batch_size, seq_len).
            actions: Latent actions of shape (batch_size, seq_len-1, latent_action_dim).
            
        Returns:
            Logits of shape (batch_size, seq_len-1, vocab_size).
        """
        B, T = tokens.size()
        # embed the tokens using the decoder
        # the decoder is a GPT and for encoder we mean embedding + self attention  
        token_emb = self.decoder.encoder(tokens) # (B, T-1, model_dim)
        # add the latent actions embedding 
        tokens = token_emb + actions # (B, T-1, model_dim)
        logits = self.decoder.lm_head(tokens)
        return logits

    def forward(self, tokens) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass: encode to actions then decode.
        
        Args:
            tokens: Token indices of shape (batch_size, seq_len).
            
        Returns:
            Tuple containing:
                - logits: Token predictions of shape (batch_size, seq_len-1, vocab_size).
                - actions: Quantized latent actions.
                - vq_loss: Vector quantization loss.
        """
        actions, vq_loss, _ = self.vq_encode(tokens)
        logits = self.decode(tokens[..., :-1], actions) # exclude the last token for prediction (B, T-1)
        return logits, actions, vq_loss

class LongHorizonLatentActionModel(LatentActionModel):
    """Long-horizon variant of Latent Action Model (not yet implemented).
    
    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.
        
    Raises:
        NotImplementedError: This model is not yet implemented.
    """
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("This model is not implemented yet.")

class DynamicsModel(SerialisableModule):
    """Dynamics model that predicts tokens given actions.
    
    Takes token sequences and latent actions as input and predicts next tokens.
    Used in combination with LatentActionModel for controllable generation.
    
    Args:
        vocab_size: Size of the vocabulary.
        n_layer: Number of transformer layers.
        n_head: Number of attention heads.
        n_embd: Embedding dimension.
        dropout: Dropout probability.
        bias: Whether to use bias in linear layers.
        context_length: Maximum sequence length.
        lm_head_out_size: Output size of the language model head.
        pad_token_id: Token ID for padding.
        bos_token_id: Token ID for beginning of sequence.
        eos_token_id: Token ID for end of sequence.
    """
    def __init__(self, 
                 vocab_size: int = 2048,
                 n_layer: int = 6,
                 n_head: int = 6,
                 n_embd: int = 384,
                 dropout: float = 0.2,
                 bias: bool = False,
                 context_length: int = 128,
                 lm_head_out_size: int = 2048,
                 pad_token_id: int = 0,
                 bos_token_id: int = 3,
                 eos_token_id: int = 4):
        super().__init__()

        t_conf = TransformerConfig(
            vocab_size=vocab_size,
            n_layer=n_layer,
            n_head=n_head,
            n_embd=n_embd,
            dropout=dropout,
            bias=bias,
            context_length=context_length,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
        )
        kwargs_gpt = t_conf.__dict__
        kwargs_gpt['lm_head_out_size'] = lm_head_out_size

        self.decoder = GPT(**kwargs_gpt)
        for k, v in t_conf.__dict__.items():
            setattr(self, k, v)

    def forward(self, tokens: torch.Tensor, actions: torch.Tensor):
        """Predict tokens conditioned on actions.
        
        Args:
            tokens: Token indices of shape (batch_size, seq_len).
            actions: Latent actions of shape (batch_size, seq_len-1, action_dim).
            
        Returns:
            Logits of shape (batch_size, seq_len-1, vocab_size).
        """
        B, T = tokens.size()
        token_emb = self.decoder.encoder(tokens) # (B, T-1, model_dim)
        inputs = token_emb + actions
        logits = self.decoder.lm_head(inputs)
        return logits

class ControllableGPT(SerialisableModule):
    """Controllable GPT combining Latent Action Model and Dynamics Model.
    
    Learns discrete latent action representations and uses them for controllable
    molecular generation. Combines two models: one that learns to encode actions
    from sequences, and another that generates sequences from actions.
    
    Args:
        vocab_size: Size of the vocabulary.
        n_layer: Number of transformer layers.
        n_head: Number of attention heads.
        n_embd: Embedding dimension.
        dropout: Dropout probability.
        bias: Whether to use bias in linear layers.
        context_length: Maximum sequence length.
        lm_head_out_size: Output size of the language model head.
        pad_token_id: Token ID for padding.
        bos_token_id: Token ID for beginning of sequence.
        eos_token_id: Token ID for end of sequence.
        latent_action_dim: Dimension of the latent action space.
        num_latents: Number of discrete latent codes.
        entropy_weight: Weight for entropy regularization.
        vq_beta: Beta parameter for VQ commitment loss.
    """
    def __init__(self, 
                 vocab_size: int = 2048,
                 n_layer: int = 6,
                 n_head: int = 6,
                 n_embd: int = 384,
                 dropout: float = 0.2,
                 bias: bool = False,
                 context_length: int = 128,
                 lm_head_out_size: int = 2048,
                 pad_token_id: int = 0,
                 bos_token_id: int = 3,
                 eos_token_id: int = 4,
                 latent_action_dim: int = 64,
                 num_latents: int = 512,
                 entropy_weight: float = 0.01,
                 vq_beta: float = 0.25):

                 
        super().__init__()
        lam_config = LatentActionModelConfig(vocab_size=vocab_size,
                                     n_layer=n_layer,
                                     n_head=n_head,
                                     n_embd=n_embd,
                                     dropout=dropout,
                                     bias=bias,
                                     context_length=context_length,
                                     lm_head_out_size=lm_head_out_size,
                                     pad_token_id=pad_token_id,
                                     bos_token_id=bos_token_id,
                                     eos_token_id=eos_token_id,
                                     latent_action_dim=latent_action_dim,
                                     num_latents=num_latents,
                                     entropy_weight=entropy_weight,
                                     vq_beta=vq_beta)
        
        self.lam = LatentActionModel(**lam_config.__dict__)

        self.dynamics_model = DynamicsModel(vocab_size=vocab_size,
                                           n_layer=n_layer,
                                           n_head=n_head,
                                           n_embd=n_embd,
                                           dropout=dropout,
                                           bias=bias,
                                           context_length=context_length,
                                           lm_head_out_size=lm_head_out_size,
                                           pad_token_id=pad_token_id,
                                           bos_token_id=bos_token_id,
                                           eos_token_id=eos_token_id)
        
        for k,v in lam_config.__dict__.items():
            setattr(self, k, v)

    def forward(
        self, tokens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through both LAM and dynamics model.
        
        Args:
            tokens: Token indices of shape (batch_size, seq_len).
            
        Returns:
            Tuple containing:
                - lam_logits: Predictions from latent action model.
                - dynamics_model_logits: Predictions from dynamics model.
                - vq_loss: Vector quantization loss.
        """
        lam_logits, actions, vq_loss = self.lam(tokens)

        dynamics_model_logits= self.dynamics_model(
            tokens[..., :-1], actions.detach()
        )

        return lam_logits, dynamics_model_logits, vq_loss

class PolicyNetwork(GPT):
    def __init__(self, 
                 vocab_size: int = 2048,
                 n_layer: int = 6,
                 n_head: int = 6,
                 n_embd: int = 384,
                 dropout: float = 0.2,
                 bias: bool = False,
                 context_length: int = 128,
                 lm_head_out_size: int = 2048,
                 pad_token_id: int = 0,
                 bos_token_id: int = 3,
                 eos_token_id: int = 4):
        
        super().__init__(vocab_size=vocab_size,
                         n_layer=n_layer,
                         n_head=n_head,
                         n_embd=n_embd,
                         dropout=dropout,
                         bias=bias,
                         context_length=context_length,
                         lm_head_out_size=lm_head_out_size,
                         pad_token_id=pad_token_id,
                         bos_token_id=bos_token_id,
                         eos_token_id=eos_token_id)
    @torch.no_grad()
    def generate(
        self,
        dynamics_model: DynamicsModel,
        lam: LatentActionModel,
        context: torch.Tensor | None = None,      # (B,T) or (T,)
        n_mols: int = 256,
        max_new_tokens: int = 128,                # total final length cap, not "to add"
        temperature: float = 1.0,
        top_k: int | None = None,
    ):
        device = self.lm_head.weight.device

        # --- Prepare initial context (B, T) ---
        if context is None:
            B = n_mols
            context = torch.full((B, 1), self.bos_token_id, dtype=torch.long, device=device)
            prev_actions = None
            # no previous actions because remember (s_t,a_t) -> s_{t+1} and for t=0 we still need to decide the action
        else:
            if context.dim() == 1:
                context = context.unsqueeze(0)  # (1, T)
            context = context.to(device=device, dtype=torch.long)
            # here instead we have states from s_0, ...,s_t and we need actions from a_0, ..., a_{t-1},
            # the model is going to predict a_t and then the dynamics model s_{t+1}
            prev_actions, _, _ = lam.vq_encode(context)
            

        # We will grow until either EOS everywhere or length hits max_new_tokens
        T0 = context.size(1)

        # for context len states there are context_len - 1 actions
        target_len = min(max_new_tokens, self.config.context_length - 1)  

        # Track which rows are finished
        finished = torch.zeros(context.size(0), dtype=torch.bool, device=device)

        # Keep generating while there is room and not all finished
        while context.size(1) < target_len and not torch.all(finished):
            # Crop to context_length if needed
            idx_cond = context if context.size(1) <= self.config.context_length else context[:, -self.config.context_length:]

            # Forward pass -> logits for all positions
            logits, _, = self(idx_cond)                     # (B, T_ctx, n_actions)
            logits = logits[:, -1, :]                         # (B, n_actions)

            next_action = sample_from_logits(tensor_logits=logits, temperature=temperature, top_k=top_k)  # (B,1)

            if prev_actions is None:
                actions = next_action.unsqueeze(1)  # (B,1,1)
            else:
                actions = torch.cat([prev_actions, next_action.unsqueeze(1)], dim=1)  # (B,t,1)

            # Predict next state using dynamics model
            logits_dm = dynamics_model(context, actions)  # (B, t, vocab)
            logits_dm = logits_dm[:, -1, :]  # (B, vocab)
            next_state = sample_from_logits(tensor_logits=logits_dm, temperature=temperature, top_k=top_k)  # (B,1)
            # Force already-finished rows to keep emitting EOS (no changes after EOS)
            eos = torch.full_like(next_state, self.eos_token_id)
            next_state = torch.where(finished.unsqueeze(1), eos, next_state)           # (B,1)

            # Update finished mask for rows that just produced EOS
            finished = finished | (next_state.squeeze(1) == self.eos_token_id)

            # Append
            context = torch.cat([context, next_state], dim=1)
        return context


# TODO(epignatelli): this model is unclear, what is it?
# class GPTwithValueHead(GPT):
#     def __init__(self, config: TransformerConfig, tokenizer_path: str):
#         self.value_head = nn.Linear(self.model.config.n_embd, 1)

#     def forward(self, input_ids: torch.Tensor, idxs: torch.Tensor):
#         B, _ = input_ids.shape
#         _, _, hidden_states = self.encoder(input_ids)
#         logits = self.lm_head(hidden_states)  # shape: (B, N, vocab_size)
#         logits = logits[torch.arange(B), idxs]
#         hidden_states = hidden_states[torch.arange(B), idxs]
#         values = self.value_head(hidden_states)
#         return logits, values

#     @torch.no_grad()
#     def generate(
#         self,
#         input_ids: torch.Tensor,
#         max_length: int | None = None,
#     ) -> torch.Tensor:
#         """
#         Sample one **batch** of sequences from the model.

#         Args
#         ----
#         input_ids : (batch, seq_len) starting tokens (already on the correct device)
#         max_length: maximum total length for each generated sequence

#         Returns
#         -------
#         torch.Tensor  # shape (batch, ≤ max_length)
#         """
#         assert self.model is not None, "Model must be initialized before generation."
#         if max_length is None:
#             max_length = self.hparams.context_length

#         generated = input_ids.clone().to(self.device)  # (B, L0)
#         batch_size, cur_len = generated.shape

#         # For each item in the batch, store the index of the *last* token we just fed in
#         idx = torch.full(
#             (batch_size,), cur_len - 1, dtype=torch.long, device=self.device
#         )

#         # Track which sequences have already finished (hit EOS)
#         finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

#         # Generate until everyone is finished or we hit the length limit
#         for _ in range(max_length - cur_len):
#             # Forward expects (B, seq_len) and a per‑row index tensor
#             logits, _ = self.forward(generated, idxs=idx)  # (B, vocab)
#             probs = torch.softmax(logits, dim=-1)  # (B, vocab)

#             # Sample one token per sequence → shape (B, 1)
#             next_token = torch.distributions.Categorical(probs).sample().unsqueeze(1)

#             # Append and update bookkeeping
#             generated = torch.cat([generated, next_token], dim=1)
#             newly_finished = next_token.squeeze(1) == self.hparams.eos_token_id
#             finished |= newly_finished

#             if finished.all():
#                 break

#             # Advance idx only for unfinished sequences; finished rows keep any value
#             idx = idx + (~finished).long()

#         return generated


# TODO(epignatelli): this model is unclear, what is it? What's the difference with GPTwithValueHead?
# Shouldn't this be just a PPO agent?
#
# class TransformerPolicy(SerialisableModule):
#     def __init__(self, enc: GPT, n_actions: int, hparams: PPOHparams = None):
#         self.enc = enc
#         self.config = self.enc.config
#         self.n_actions = n_actions
#         self.hparams = load_hparams() if hparams is None else hparams
#         self.action_head = nn.Linear(self.config.n_embd, self.n_actions)
#         self.value_head = nn.Linear(self.config.n_embd, 1)
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#         _, _, x = self.enc(x)  # shape: (B, T, n_embd)
#         action_logits = self.action_head(x)  # shape: (B, T, n_actions)
#         values = self.value_head(x)  # shape: (B, T, 1)
#         return action_logits, values

#     @property
#     def embedding_dim(self) -> int:
#         return self.config.n_embd

#     @torch.no_grad()
#     def generate(self, states: torch.Tensor, max_length: int = 100, **kwargs):
#         if states is None:
#             batch_size = 1000
#             states = torch.full(
#                 (batch_size,),
#                 self.hparams.SOS_TOKEN,
#                 dtype=torch.long,
#                 device=self.device,
#             )
#             cur_len = 1
#         else:
#             batch_size, cur_len = states.shape

#         lam = kwargs.get("lam", None)
#         env = kwargs.get("env", None)
#         if lam is None or env is None:
#             raise ValueError("Both 'lam' and 'env' must be provided for generation.")
#         n_envs = env.n_envs
#         env.n_envs = batch_size
#         env.reset_all()

#         if (
#             cur_len > 1
#         ):  # if the state contains more than the initial state i can calculate the past actions
#             input_ids = states.to(self.device)
#             out = lam.vq_encode(input_ids, training=False)
#             past_action_idxs = out["indices"]  # shape: (1, T-1)
#             prompts = [
#                 {"states": input_id, "actions": past_action_idx}
#                 for input_id, past_action_idx in zip(input_ids, past_action_idxs)
#             ]
#             env.reset_all(prompts=prompts)
#         else:
#             # if the state is just the initial state the env is already reseted by env.reset_all()
#             pass
#         idx = torch.full(
#             (batch_size,), cur_len - 1, dtype=torch.long, device=self.device
#         )

#         # Track which sequences have already finished (hit EOS)
#         finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

#         for _ in range(max_length - cur_len):
#             states = env._observation().to(self.device)
#             logits, _ = self(states=states, idxs=idx)
#             dist = Categorical(logits=logits)
#             actions = dist.sample()
#             next_states = env.step(actions)
#             newly_finished = torch.Tensor(
#                 [next_state.terminated for next_state in next_states]
#             ).to(self.device)
#             finished |= newly_finished
#             if finished.all():
#                 break

#             idx = idx + (~finished).long()
#         env.n_envs = n_envs
#         return env._observation()

#     # TODO(epignatelli): why is this method here? it is identical to the one in GPT and LatentActionModel
#     def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
#         # start with all of the candidate parameters
#         param_dict = {pn: p for pn, p in self.named_parameters()}
#         # filter out those that do not require grad
#         param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
#         # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
#         # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
#         decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
#         nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
#         optim_groups = [
#             {"params": decay_params, "weight_decay": weight_decay},
#             {"params": nodecay_params, "weight_decay": 0.0},
#         ]
#         num_decay_params = sum(p.numel() for p in decay_params)
#         num_nodecay_params = sum(p.numel() for p in nodecay_params)
#         print(
#             f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
#         )
#         print(
#             f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
#         )
#         # Create AdamW optimizer and use the fused version if it is available
#         fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
#         use_fused = fused_available and device_type == "cuda"
#         extra_args = dict(fused=True) if use_fused else dict()
#         optimizer = torch.optim.AdamW(
#             optim_groups, lr=learning_rate, betas=betas, **extra_args
#         )
#         print(f"using fused AdamW: {use_fused}")

#         return optimizer

# TODO(epignatelli): this model is unclear, what is it?
# class Decoder(GPT):
#     def __init__(self, config: TransformerConfig, tokenizer_path: str):
#         self.pair_head = PairwiseHead(
#             hidden_dim=config.n_embd,
#             vocab_size=config.vocab_size,
#             max_len=config.context_length,
#             rel_pos_dim=32,
#         )

#     def forward(
#         self,
#         tokens: torch.Tensor,
#         actions: torch.Tensor,
#         learn_high_level_actions: bool = False,
#     ):

#         B, T = tokens.size()
#         inputs = tokens[:, :-1]  # exclude the last token for prediction
#         targets = tokens[:, 1:]
#         tok_emb = self.encoder.wte(inputs)
#         inputs = tok_emb + actions
#         assert (
#             T <= self.config.context_length
#         ), f"Cannot forward sequence of length {T}, block size is only {self.config.context_length}"
#         pos = torch.arange(0, T - 1, dtype=torch.long, device=self.device)  # shape (t)
#         pos_emb = self.encoder.wpe(pos)
#         x = self.encoder.drop(inputs + pos_emb)
#         for block in self.encoder.h:
#             x = block(x)
#         x = self.encoder.ln_f(x)

#         if learn_high_level_actions:
#             logits = self.pair_head(x)
#             loss = self.pairwise_future_ce_loss(
#                 logits=logits, targets=targets, pad_id=0
#             )

#         else:
#             logits = self.lm_head(x)
#             loss = F.cross_entropy(
#                 logits.view(-1, logits.size(-1)),
#                 targets.reshape(
#                     -1,
#                 ),
#                 ignore_index=0,
#             )  # 0 is the padding token

#         return logits, loss, x

#     def pairwise_future_ce_loss(
#         self,
#         logits: torch.Tensor,  # (B, T_src, T_tgt, V)
#         targets: torch.Tensor,  # (B,         T_tgt)
#         pad_id: int | None = 0,  # set None if no padding
#     ):
#         """
#         Cross entropy on the upper diagonal only, i.e. each source
#         position learns to predict its own token (+self) and/or all future
#         tokens, but never the past.
#         """
#         B, T_src, T_tgt, V = logits.shape
#         assert T_src == T_tgt, "Head returns square grid with T_src == T_tgt"

#         # ---- broadcast ground‑truth along source axis --------------------------
#         gold = targets.unsqueeze(1).expand(-1, T_src, -1)  # (B, T, T)

#         # ---- build boolean mask  (B, T, T) ------------------------------
#         diag = 1
#         tri = torch.triu(
#             torch.ones(T_src, T_tgt, dtype=torch.bool, device=logits.device),
#             diagonal=diag,
#         )
#         mask = tri.unsqueeze(0).expand(B, -1, -1)  # (B, T, T)

#         if pad_id is not None:
#             pad_mask = gold.eq(pad_id)  # remove PAD targets
#             mask = mask & (~pad_mask)

#         # ---- select only valid pairs ------------------------------------------
#         logits_sel = logits[mask]  # shape (*, V)
#         targets_sel = gold[mask]  # shape (*,)

#         loss = F.cross_entropy(logits_sel, targets_sel)
#         return loss


# class GPTEncoderForRL(nn.Module):
#     """Wrapper around a pretrained GPT model that exposes only encoder outputs.
    
#     This wrapper is designed for use with PPO-based RL training. It takes token
#     sequences as observations and returns embeddings at the last non-padding
#     position, suitable for policy and value head inputs.
    
#     The PPO algorithm expects:
#         - Input: observation tensor of shape (batch, seq_len) containing token IDs
#         - Output: embedding tensor of shape (batch, n_embd)
    
#     Args:
#         gpt_model: A pretrained GPT model instance.
#         pad_token_id: Token ID used for padding (to find last valid position).
#     """
    
#     def __init__(self, gpt_model: GPT, pad_token_id: int = 0):
#         super().__init__()
#         self.encoder = gpt_model.encoder
#         self.pad_token_id = pad_token_id
#         self.n_embd = gpt_model.config.n_embd
#         self.config = gpt_model.config
    
#     def forward(self, observation: torch.Tensor) -> torch.Tensor:
#         """Encode token sequence and return embedding at the current position.
        
#         Args:
#             observation: Token IDs of shape (batch, seq_len).
#                         Can be int16, int32, or long dtype.
        
#         Returns:
#             Embeddings of shape (batch, n_embd) at the last non-padding position.
#         """
#         # Convert to long for embedding lookup
#         if observation.dtype != torch.long:
#             observation = observation.long()
        
#         # Get encoder hidden states: (batch, seq_len, n_embd)
#         hidden_states = self.encoder(observation)
        
#         # Find the last non-padding position for each sequence
#         mask = observation != self.pad_token_id  # (batch, seq_len)
#         seq_lengths = mask.sum(dim=1).clamp(min=1) - 1  # (batch,)
        
#         # Gather embedding at last valid position
#         batch_size = hidden_states.size(0)
#         batch_indices = torch.arange(batch_size, device=hidden_states.device)
#         embeddings = hidden_states[batch_indices, seq_lengths]  # (batch, n_embd)
        
#         return embeddings
    
#     @classmethod
#     def from_pretrained(cls, checkpoint_path: str, gpt_config: dict, pad_token_id: int = 0):
#         """Load a GPTEncoderForRL from a pretrained GPT checkpoint.
        
#         Args:
#             checkpoint_path: Path to the saved GPT model checkpoint.
#             gpt_config: Dictionary of GPT configuration parameters.
#             pad_token_id: Padding token ID.
            
#         Returns:
#             GPTEncoderForRL instance with loaded weights.
#         """
#         gpt_model = GPT(**gpt_config)
#         gpt_model.load(checkpoint_path)
#         return cls(gpt_model, pad_token_id=pad_token_id)