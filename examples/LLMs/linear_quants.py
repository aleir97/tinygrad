from tinygrad.tensor import Tensor
from tinygrad.helpers import dtypes
import gguf
from gguf.constants import GGML_QUANT_SIZES, GGMLQuantizationType
import numpy as np
from sentencepiece import SentencePieceProcessor, sentencepiece_model_pb2
from tinygrad.jit import TinyJit

class AbsmaxQuantizedLinear:
  def __init__(self, in_features, out_features, bias=False):
    assert bias == False
    self.weight = Tensor.ones(out_features, in_features, dtype=dtypes.int8)
    self.scale = Tensor.ones(out_features, dtype=dtypes.half)

  def __call__(self, x):
    return x.dot(self.weight.cast(dtype=dtypes.half).T*self.scale)

  @staticmethod
  def quantize(tensors):
    new_tensors = {}
    for name,v in tensors.items():
      if "feed_forward" in name or ("attention.w") in name or name == "output.weight":
        scale = v.abs().max(axis=1) / 127.0
        int8_weight = (v.T/scale).T.cast(dtype=dtypes.int8)
        new_tensors[name] = int8_weight
        new_tensors[name.replace('weight', 'scale')] = scale
      else:
        new_tensors[name] = v
    return new_tensors

class tinyGGUF:
  def __init__(self, fn:str):
    self.reader = gguf.GGUFReader(fn)

  def load_gguf_weights(self, model):
    sd = {}
    gguf_to_tinygrad_keymap = {
        'token_embd.weight': 'tok_embeddings.weight',
        **{f"blk.{i}.attn_norm.weight": f"layers.{i}.attention_norm.weight" for i in range(len(model.layers))},
        **{f"blk.{i}.attn_{v}.weight": f"layers.{i}.attention.w{v[0]}.weight" for v in ["q", "k", "v", "output"] for i in range(len(model.layers))},
        **{f"blk.{i}.ffn_norm.weight": f"layers.{i}.ffn_norm.weight" for i in range(len(model.layers))},
        **{f"blk.{i}.ffn_{x}.weight": f"layers.{i}.feed_forward.w{y}.weight" for x,y in {"gate": 1, "down": 2, "up": 3}.items() for i in range(len(model.layers))},
        'output_norm.weight': 'norm.weight', 'output.weight': 'output.weight',
    }
    for tensor in self.reader.tensors:
      scale = None
      k = gguf_to_tinygrad_keymap[tensor.field.name]
      if tensor.tensor_type == GGMLQuantizationType.Q4_0:
        if 'embedding' not in k:
          w, scale = QK4_0Linear.get_weight_and_scale_from_q4_0(tensor.data)
        else:
          w = QK4_0Linear.dequantize_q4_0(tensor)
      elif tensor.tensor_type == GGMLQuantizationType.Q6_K:
        w = QK4_0Linear.dequantize_q6_k(tensor)
      elif tensor.tensor_type == GGMLQuantizationType.F32:
        w = Tensor(tensor.data).reshape(np.flip(tensor.shape).tolist()).half()
      else: raise RuntimeError("Quantization type still not supported!")

      sd[k] = w
      if scale is not None:
        sd[k.replace('.weight', '.scale')] = scale
    return sd

  def load_gguf_tokenizer(self):
    tokens = [str(bytes(self.reader.fields['tokenizer.ggml.tokens'].parts[idx]), encoding="utf-8") for idx in self.reader.fields['tokenizer.ggml.tokens'].data]
    scores = [pv for idx in self.reader.fields['tokenizer.ggml.scores'].data for pv in self.reader.fields['tokenizer.ggml.scores'].parts[idx].tolist()]
    types  = [pv for idx in self.reader.fields['tokenizer.ggml.token_type'].data for pv in self.reader.fields['tokenizer.ggml.token_type'].parts[idx].tolist()]

    # Model tokens for Sentence Piece use Google's Protocol Buffer
    token_model = sentencepiece_model_pb2.ModelProto()
    for i in range(len(tokens)):
      token = token_model.pieces.add()
      token.piece = tokens[i]
      token.score = scores[i]
      token.type  = types[i]
      if token.type == gguf.TokenType.BYTE:
        token_model.trainer_spec.byte_fallback = 1

    token_model.trainer_spec.unk_id = self.reader.fields['tokenizer.ggml.unknown_token_id'].parts[-1][0]
    token_model.trainer_spec.bos_id = self.reader.fields['tokenizer.ggml.bos_token_id'].parts[-1][0]
    token_model.trainer_spec.eos_id = self.reader.fields['tokenizer.ggml.eos_token_id'].parts[-1][0]
    # Load the model from the Protocol Buffer created with the .gguf info
    sp = SentencePieceProcessor()
    sp.load_from_serialized_proto(token_model.SerializeToString())
    return sp

# GMML Q4_0 Quantification
class QK4_0Linear:
  def __init__(self, in_features, out_features, bias=False):
    assert bias == False
    self.in_features = in_features
    self.out_features = out_features
    dim = out_features * in_features
    # each block stores 32 weights
    assert dim % 32 == 0
    n_blocks = dim // 32
    self.weight = Tensor.ones(n_blocks, 16, dtype=dtypes.uint8)
    self.scale = Tensor.ones(n_blocks, 1, dtype=dtypes.half)

  @staticmethod
  def quantize(tensors):
    # https://github.com/ggerganov/llama.cpp/blob/master/ggml-quants.c#L427
    new_tensors = {}
    for name,v in tensors.items():
      if "feed_forward" in name or ("attention.w") in name or name == "output.weight":
        blocks = v.reshape(-1, 32)
        weight = Tensor.zeros(blocks.shape[0], 16, dtype=dtypes.uint8)
        _min, _max = blocks.min(axis=1), blocks.max(axis=1)
        scale = (_min.abs() > _max).where(_min, _max) / -8
        scale_inverse = scale.where(scale.reciprocal(), Tensor.zeros_like(scale))
        quants = (((blocks * scale_inverse.unsqueeze(1)) + 8.5).clip(0, 15)).cast(dtypes.uint8)
        weight = weight.xor(quants[:, :16])
        weight = weight.xor(quants[:, 16:] * 16)
        scale = scale.unsqueeze(1).half()
        new_tensors[name] = weight
        new_tensors[name.replace('weight', 'scale')] = scale
      else:
        new_tensors[name] = v
    return new_tensors

  def dequantize(self):
    # https://github.com/ggerganov/llama.cpp/blob/master/ggml-quants.c#L1074
    div = (self.weight / 16)
    return (
        (Tensor.cat(self.weight - (div * 16), div, dim=1).cast(dtypes.int8) - 8).half() * self.scale
      ).reshape((self.out_features, self.in_features))

  def __call__(self, x):
    return x.dot(self.dequantize().T)

  @staticmethod
  def dequantize_q4_0(tensor: gguf.ReaderTensor):
    # https://github.com/ggerganov/llama.cpp/blob/master/ggml-quants.c#L1074
    block_sz, type_sz = GGML_QUANT_SIZES[tensor.tensor_type]
    blks = tensor.data.reshape(-1,type_sz)
    scales  = Tensor(blks[:,:2].flatten().view(np.float16)).repeat((block_sz,1)).transpose().cast(dtypes.float16)
    weights = Tensor(blks)[:,2:]
    div = (weights / 16)
    return ((Tensor.cat(weights - (div * 16), div, dim=1).cast(dtypes.int8) - 8) * scales).reshape(np.flip(tensor.shape).tolist())

  @staticmethod
  def get_weight_and_scale_from_q4_0(tensor):
    blocks = tensor.reshape(-1, 18)
    weight = blocks[:, 2:]
    scale = blocks[:, :2].view(np.float16)
    return Tensor(weight), Tensor(scale)

  # This should be on a Q6k class
  @staticmethod
  def dequantize_q6_k(tensor: gguf.ReaderTensor):
    # https://github.com/ggerganov/llama.cpp/blob/master/ggml-quants.c#L2263
    k , _= GGML_QUANT_SIZES[tensor.tensor_type]
    gguf_tensor_data = tensor.data.reshape((-1, 210))
    ql = gguf_tensor_data[:, :k//2]  # Lower 4 bits, uint8
    qh = gguf_tensor_data[:, k//2:(k//2)+(k//4)]  # Upper 2 bits, uint8
    scales = gguf_tensor_data[:, (k//2)+(k//4):(k//2)+(k//4)+(k//16)].view(np.int8)  # scales, int8
    d = gguf_tensor_data[:, (k//2)+(k//4)+(k//16):].view(np.float16).astype(np.float16)  # super-block scale, fp16

    vals = []
    for n in range(2):
      q = []
      ql_idx = n*64
      qh_idx = n*32
      scales_idx = n*8
      q.append(((ql[:, ql_idx:32+ql_idx] & 0xF) | ((qh[:, qh_idx:32+qh_idx] >> 0) & 3) << 4).astype(np.int8) - 32)
      q.append(((ql[:, ql_idx+32:64+ql_idx] & 0xF) | ((qh[:, qh_idx:32+qh_idx] >> 2) & 3) << 4).astype(np.int8) - 32)
      q.append(((ql[:, ql_idx:32+ql_idx] >> 4) | ((qh[:, qh_idx:32+qh_idx] >> 4) & 3) << 4).astype(np.int8) - 32)
      q.append(((ql[:, ql_idx+32:ql_idx+64] >> 4) | ((qh[:, qh_idx:32+qh_idx] >> 6) & 3) << 4).astype(np.int8) - 32)
      for i in range(8):
        qval = q[i//2][:, :16] if i % 2 == 0 else q[i//2][:, 16:]
        vals.append(d * scales[:, i+scales_idx:i+scales_idx+1] * qval)

    y = np.concatenate(vals, axis=1).reshape(np.flip(tensor.shape))
    return Tensor(y)


