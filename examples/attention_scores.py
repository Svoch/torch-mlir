import torch
import torch.nn as nn

import torch_mlir
from torch_mlir_e2e_test.linalg_on_tensors_backends import refbackend

class AttentionScores(torch.nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super(AttentionScores, self).__init__()
        self.query = nn.Linear(embedding_dim, num_heads)
        self.key = nn.Linear(embedding_dim, num_heads)

    def forward(self, inputs):
        query = self.query(inputs)
        key = self.key(inputs)
        scores = torch.matmul(query, key.transpose(0, 1))
        return scores

# instantiate the model and define inputs 
attention_scores = AttentionScores(embedding_dim=10, num_heads=2)
inputs = torch.rand(5, 10)
# get reference forward-propagate results using torch
print("forward propagate result attention_scores is ", attention_scores.forward(inputs))

# compile to LinAlg 
linalg_module = torch_mlir.compile(attention_scores, inputs, output_type=torch_mlir.OutputType.LINALG_ON_TENSORS)
print("compiled encoder to LinAlg is:\n", linalg_module)

linalg_backend = refbackend.RefBackendLinalgOnTensorsBackend()
compiled = linalg_backend.compile(linalg_module)
jit_module = linalg_backend.load(compiled)
# check results
print("compiled for backend module is:\n", jit_module.forward(inputs.numpy()))

# compile to LinAlg
# This line fails with the error: failed to legalize operation 'torch.constant.int' 
tosa_module = torch_mlir.compile(attention_scores, inputs, output_type=torch_mlir.OutputType.TOSA)
print("compiled encoder to TOSA is:\n", tosa_module)
