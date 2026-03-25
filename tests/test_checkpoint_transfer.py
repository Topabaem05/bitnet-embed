from __future__ import annotations

import torch
import torch.nn as nn

from bitnet_embed.train.factory import TrainingCheckpointState, load_checkpoint_weights


class _BaseLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(2, 2, dtype=torch.float32))


class _QProj(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base_layer = _BaseLayer()


class _SelfAttn(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = _QProj()


class _Layer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = _SelfAttn()


class _ModelRoot(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_Layer()])


class _BaseModelWrapper(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _ModelRoot()


class _BackboneWrapper(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base_model = _BaseModelWrapper()


class _Backbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = _BackboneWrapper()


class _DummyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.plain_weight = nn.Parameter(torch.zeros(2, 2, dtype=torch.float32))
        self.backbone = _Backbone()


def test_load_checkpoint_weights_maps_plain_backbone_key_to_peft_base_layer() -> None:
    model = _DummyModel()
    checkpoint_state = TrainingCheckpointState(
        model_state={
            "backbone.backbone.layers.0.self_attn.q_proj.weight": torch.ones(2, 2),
            "plain_weight": torch.full((2, 2), 2.0),
        },
        optimizer_state={},
        scheduler_state={},
        global_step=1,
        mode="head_only",
    )

    missing, unmatched = load_checkpoint_weights(model, checkpoint_state)

    assert unmatched == []
    assert missing == []
    target_state = model.state_dict()
    assert torch.allclose(
        target_state[
            "backbone.backbone.base_model.model.layers.0.self_attn.q_proj.base_layer.weight"
        ],
        torch.ones(2, 2),
    )
    assert torch.allclose(target_state["plain_weight"], torch.full((2, 2), 2.0))
