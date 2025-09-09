import torch
from torch import nn
from torch_geometric.nn.aggr import MeanAggregation

from gatr.interface import extract_scalar
from xformers.ops.fmha import BlockDiagonalMask

from experiments.tagging.embedding import embed_tagging_data_into_ga

def xformers_sa_mask(batch, materialize=False):
    """
    Construct attention mask that makes sure that objects only attend to each other
    within the same batch element, and not across batch elements

    Parameters
    ----------
    batch: torch.tensor
        batch object in the torch_geometric.data naming convention
        contains batch index for each event in a sparse tensor
    materialize: bool
        Decides whether a xformers or ('materialized') torch.tensor mask should be returned
        The xformers mask allows to use the optimized xformers attention kernel, but only runs on gpu

    Returns
    -------
    mask: xformers.ops.fmha.attn_bias.BlockDiagonalMask or torch.tensor
        attention mask, to be used in xformers.ops.memory_efficient_attention
        or torch.nn.functional.scaled_dot_product_attention
    """
    bincounts = torch.bincount(batch).tolist()
    mask = BlockDiagonalMask.from_seqlens(bincounts)
    if materialize:
        # materialize mask to torch.tensor (only for testing purposes)
        mask = mask.materialize(shape=(len(batch), len(batch))).to(batch.device)
    return mask


class TaggingGATrWrapper(nn.Module):
    """
    L-GATr for tagging
    """

    def __init__(
        self,
        net,
        mean_aggregation=False,
        force_xformers=True,
        jet_scalar_embed_dim=10,
        jet_elements=4,
        const_elements=4,
    ):
        super().__init__()
        self.net = net
        self.aggregation = MeanAggregation() if mean_aggregation else None
        self.force_xformers = force_xformers
        self.embed_jet = nn.Linear(jet_elements, jet_scalar_embed_dim)
        self.embed_const = nn.Linear(const_elements, jet_scalar_embed_dim)

    def forward(self, batch, cfg_data):
        embedding = embed_tagging_data_into_ga(
            batch.x, batch.scalars, batch.jet_features, batch.ptr, cfg_data, self.embed_jet, self.embed_const
        )
        embedding["mv"] = embedding["mv"].to(dtype=torch.float32)
        embedding["s"] = embedding["s"].to(dtype=torch.float32)
        multivector = embedding["mv"].unsqueeze(0)
        scalars = embedding["s"].unsqueeze(0)

        mask = xformers_sa_mask(embedding["batch"], materialize=not self.force_xformers)
        if not self.force_xformers and mask.dtype != scalars.dtype:
            mask = mask.to(scalars.dtype)
        multivector_outputs, scalar_outputs = self.net(
            multivector, scalars=scalars, attention_mask=mask
        )
        logits = self.extract_from_ga(
            multivector_outputs,
            scalar_outputs,
            embedding["batch"],
            embedding["is_global"],
        )

        return logits

    def extract_from_ga(self, multivector, scalars, batch, is_global):
        outputs = extract_scalar(multivector)[0, :, :, 0]
        if self.aggregation is not None:
            logits = self.aggregation(outputs, index=batch)
        else:
            logits = outputs[is_global]
        return logits
