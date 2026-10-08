from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


VALID_DISTANCE_METRICS = (
    "euclidean",
    "cosine",
)


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups

    return 1


class ConvBlock(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
    ) -> None:
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                input_channels,
                output_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(
                num_groups=_group_count(output_channels),
                num_channels=output_channels,
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

    def forward(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:
        return self.block(images)


class ProtoNetEncoder(nn.Module):
    architecture_id = "conv4_groupnorm_v2"

    def __init__(
        self,
        input_channels: int = 3,
        hidden_channels: int = 64,
        embedding_dim: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if input_channels <= 0:
            raise ValueError(
                "input_channels must be positive"
            )

        if hidden_channels <= 0:
            raise ValueError(
                "hidden_channels must be positive"
            )

        if embedding_dim <= 0:
            raise ValueError(
                "embedding_dim must be positive"
            )

        if not 0.0 <= dropout < 1.0:
            raise ValueError(
                "dropout must be in [0, 1)"
            )

        self.features = nn.Sequential(
            ConvBlock(
                input_channels,
                hidden_channels,
            ),
            ConvBlock(
                hidden_channels,
                hidden_channels,
            ),
            ConvBlock(
                hidden_channels,
                hidden_channels,
            ),
            ConvBlock(
                hidden_channels,
                hidden_channels,
            ),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)

        if embedding_dim == hidden_channels:
            self.projection = nn.Identity()
        else:
            self.projection = nn.Linear(
                hidden_channels,
                embedding_dim,
            )

    def forward(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:
        if images.ndim != 4:
            raise ValueError(
                "Expected BCHW image tensor, received "
                f"{tuple(images.shape)}"
            )

        features = self.features(images)
        features = self.pool(features)
        features = torch.flatten(
            features,
            start_dim=1,
        )
        features = self.dropout(features)

        return self.projection(features)


class ProtoNet(nn.Module):
    architecture_id = ProtoNetEncoder.architecture_id

    def __init__(
        self,
        input_channels: int = 3,
        hidden_channels: int = 64,
        embedding_dim: int = 64,
        dropout: float = 0.0,
        distance_metric: str = "euclidean",
        temperature: float = 1.0,
    ) -> None:
        super().__init__()

        if distance_metric not in VALID_DISTANCE_METRICS:
            raise ValueError(
                f"Unsupported distance metric: "
                f"{distance_metric!r}"
            )

        if temperature <= 0:
            raise ValueError(
                "temperature must be positive"
            )

        self.distance_metric = distance_metric
        self.temperature = float(temperature)

        self.encoder = ProtoNetEncoder(
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            embedding_dim=embedding_dim,
            dropout=dropout,
        )

    def forward(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:
        return self.encoder(images)

    def episode_logits(
        self,
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
        query_images: torch.Tensor,
        n_way: int,
    ) -> torch.Tensor:
        if n_way <= 1:
            raise ValueError(
                "n_way must be greater than one"
            )

        support_embeddings = self.encoder(
            support_images
        )

        query_embeddings = self.encoder(
            query_images
        )

        if self.distance_metric == "cosine":
            support_embeddings = F.normalize(
                support_embeddings,
                p=2,
                dim=1,
            )

            query_embeddings = F.normalize(
                query_embeddings,
                p=2,
                dim=1,
            )

        prototypes: list[torch.Tensor] = []

        for class_index in range(n_way):
            class_mask = (
                support_labels == class_index
            )

            if not torch.any(class_mask):
                raise ValueError(
                    f"No support samples for class "
                    f"{class_index}"
                )

            prototypes.append(
                support_embeddings[
                    class_mask
                ].mean(dim=0)
            )

        prototype_tensor = torch.stack(
            prototypes,
            dim=0,
        )

        if self.distance_metric == "cosine":
            prototype_tensor = F.normalize(
                prototype_tensor,
                p=2,
                dim=1,
            )

            similarities = (
                query_embeddings
                @ prototype_tensor.transpose(0, 1)
            )

            return (
                similarities
                / self.temperature
            )

        squared_distances = (
            query_embeddings[:, None, :]
            - prototype_tensor[None, :, :]
        ).pow(2).sum(dim=-1)

        return (
            -squared_distances
            / self.temperature
        )
