from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence

import cv2
import numpy as np
import torch
from skimage.feature import local_binary_pattern
from torch import nn


EPSILON = 1e-12


PIXEL_FEATURE_NAMES = (
    "pixel_channel_0_mean",
    "pixel_channel_1_mean",
    "pixel_channel_2_mean",
    "pixel_channel_0_std",
    "pixel_channel_1_std",
    "pixel_channel_2_std",
    "pixel_global_mean",
    "pixel_global_std",
    "pixel_entropy",
    "pixel_skewness",
    "pixel_excess_kurtosis",
    "pixel_dynamic_range",
    "pixel_contrast_ratio",
    "pixel_saturation_fraction",
)

TEXTURE_FEATURE_NAMES = (
    "texture_gradient_magnitude_mean",
    "texture_gradient_magnitude_std",
    "texture_gradient_magnitude_p90",
    "texture_sobel_x_absolute_mean",
    "texture_sobel_y_absolute_mean",
    "texture_edge_density",
    "texture_laplacian_absolute_mean",
    "texture_laplacian_std",
    "texture_laplacian_variance",
    "texture_high_frequency_energy",
    "texture_gabor_0_mean",
    "texture_gabor_0_std",
    "texture_gabor_45_mean",
    "texture_gabor_45_std",
    "texture_gabor_90_mean",
    "texture_gabor_90_std",
    "texture_gabor_135_mean",
    "texture_gabor_135_std",
    "texture_tamura_coarseness_approx",
    "texture_tamura_contrast",
    "texture_tamura_directionality",
    "texture_lbp_uniformity",
)

GEOMETRY_FEATURE_NAMES = (
    "geometry_intra_class_variance_mean",
    "geometry_intra_class_variance_std",
    "geometry_intra_class_variance_min",
    "geometry_intra_class_variance_max",
    "geometry_inter_prototype_distance_mean",
    "geometry_inter_prototype_distance_std",
    "geometry_inter_prototype_distance_min",
    "geometry_inter_prototype_distance_max",
    "geometry_nearest_prototype_distance_mean",
    "geometry_nearest_prototype_distance_std",
    "geometry_prototype_norm_mean",
    "geometry_prototype_norm_std",
    "geometry_support_embedding_norm_mean",
    "geometry_support_embedding_norm_std",
    "geometry_fisher_ratio",
    "geometry_cluster_compactness",
    "geometry_centroid_separation",
    "geometry_separation_to_variance_ratio",
)

SPECTRAL_FEATURE_NAMES = (
    "spectral_explained_variance_ratio_1",
    "spectral_explained_variance_ratio_2",
    "spectral_explained_variance_ratio_3",
    "spectral_explained_variance_ratio_top5",
    "spectral_explained_variance_ratio_top10",
    "spectral_effective_rank",
    "spectral_normalized_entropy",
    "spectral_flatness",
    "spectral_anisotropy",
    "spectral_log_condition_number",
)

ZERO_FEATURE_NAMES = (
    PIXEL_FEATURE_NAMES
    + TEXTURE_FEATURE_NAMES
    + GEOMETRY_FEATURE_NAMES
    + SPECTRAL_FEATURE_NAMES
)

assert len(PIXEL_FEATURE_NAMES) == 14
assert len(TEXTURE_FEATURE_NAMES) == 22
assert len(GEOMETRY_FEATURE_NAMES) == 18
assert len(SPECTRAL_FEATURE_NAMES) == 10
assert len(ZERO_FEATURE_NAMES) == 64


DATASET_NORMALIZATION = {
    "omniglot": (
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5),
    ),
    "cifar100": (
        (0.5071, 0.4867, 0.4408),
        (0.2675, 0.2565, 0.2761),
    ),
    "miniimagenet": (
        (0.485, 0.456, 0.406),
        (0.229, 0.224, 0.225),
    ),
    "dtd": (
        (0.485, 0.456, 0.406),
        (0.229, 0.224, 0.225),
    ),
    "flowers102": (
        (0.485, 0.456, 0.406),
        (0.229, 0.224, 0.225),
    ),
}


def _normalise_dataset_name(dataset_name: str) -> str:
    normalized = (
        dataset_name
        .strip()
        .lower()
        .replace("-", "")
        .replace("_", "")
    )

    aliases = {
        "omniglot": "omniglot",
        "cifar100": "cifar100",
        "miniimagenet": "miniimagenet",
        "dtd": "dtd",
        "flowers102": "flowers102",
    }

    if normalized not in aliases:
        raise ValueError(
            f"No descriptor normalization registered for "
            f"{dataset_name!r}"
        )

    return aliases[normalized]


def denormalize_to_unit_interval(
    images: torch.Tensor,
    dataset_name: str,
) -> torch.Tensor:
    """
    Convert normalized BCHW images back to the [0, 1] range.

    Pixel and texture descriptors must be calculated from comparable
    intensity ranges rather than dataset-specific normalized values.
    """
    if images.ndim != 4:
        raise ValueError(
            f"Expected BCHW tensor, received shape {tuple(images.shape)}"
        )

    if images.shape[1] != 3:
        raise ValueError(
            f"Expected three channels, received {images.shape[1]}"
        )

    normalized_name = _normalise_dataset_name(dataset_name)
    mean_values, std_values = DATASET_NORMALIZATION[normalized_name]

    mean = torch.tensor(
        mean_values,
        dtype=images.dtype,
        device=images.device,
    ).view(1, 3, 1, 1)

    std = torch.tensor(
        std_values,
        dtype=images.dtype,
        device=images.device,
    ).view(1, 3, 1, 1)

    return (images * std + mean).clamp(0.0, 1.0)


class FrozenDescriptorEncoder(nn.Module):
    """
    Deterministically initialized shallow encoder.

    The encoder is never trained. The same frozen weights must be used for
    every dataset and task to make geometric descriptors comparable.
    """

    def __init__(
        self,
        seed: int = 20260801,
    ) -> None:
        super().__init__()

        self.seed = int(seed)

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed)

            self.network = nn.Sequential(
                nn.Conv2d(
                    3,
                    32,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                nn.GroupNorm(8, 32),
                nn.GELU(),
                nn.AvgPool2d(2),

                nn.Conv2d(
                    32,
                    64,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                nn.GroupNorm(8, 64),
                nn.GELU(),
                nn.AvgPool2d(2),

                nn.Conv2d(
                    64,
                    64,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                nn.GroupNorm(8, 64),
                nn.GELU(),

                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
            )

        for parameter in self.parameters():
            parameter.requires_grad_(False)

        self.eval()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # Map [0, 1] input to approximately [-1, 1].
        return self.network(images * 2.0 - 1.0)


def descriptor_encoder_sha256(
    encoder: nn.Module,
) -> str:
    digest = hashlib.sha256()

    for name, tensor in sorted(encoder.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(
            tensor.detach()
            .cpu()
            .contiguous()
            .numpy()
            .tobytes()
        )

    return digest.hexdigest()


def _distribution_moments(
    values: np.ndarray,
) -> tuple[float, float]:
    flattened = values.astype(np.float64, copy=False).reshape(-1)

    mean = float(np.mean(flattened))
    std = float(np.std(flattened))

    if std <= EPSILON:
        return 0.0, 0.0

    standardized = (flattened - mean) / std

    skewness = float(np.mean(standardized ** 3))
    excess_kurtosis = float(
        np.mean(standardized ** 4) - 3.0
    )

    return skewness, excess_kurtosis


def _entropy(
    values: np.ndarray,
    bins: int = 64,
) -> float:
    histogram, _ = np.histogram(
        values,
        bins=bins,
        range=(0.0, 1.0),
    )

    probabilities = histogram.astype(np.float64)
    total = probabilities.sum()

    if total <= 0:
        return 0.0

    probabilities /= total
    probabilities = probabilities[probabilities > 0]

    return float(
        -np.sum(probabilities * np.log(probabilities))
    )


def extract_pixel_features(
    unit_images: np.ndarray,
) -> np.ndarray:
    """
    unit_images: B x 3 x H x W in [0, 1].
    """
    if unit_images.ndim != 4 or unit_images.shape[1] != 3:
        raise ValueError(
            f"Expected Bx3xHxW, received {unit_images.shape}"
        )

    channel_means = unit_images.mean(axis=(0, 2, 3))
    channel_stds = unit_images.std(axis=(0, 2, 3))

    flattened = unit_images.reshape(-1).astype(
        np.float64,
        copy=False,
    )

    global_mean = float(flattened.mean())
    global_std = float(flattened.std())

    skewness, excess_kurtosis = _distribution_moments(
        flattened
    )

    minimum = float(np.min(flattened))
    maximum = float(np.max(flattened))
    dynamic_range = maximum - minimum

    percentile_5 = float(np.percentile(flattened, 5))
    percentile_95 = float(np.percentile(flattened, 95))

    contrast_ratio = (
        percentile_95 - percentile_5
    ) / (
        percentile_95 + percentile_5 + EPSILON
    )

    saturation_fraction = float(
        np.mean(
            (flattened <= 0.01)
            | (flattened >= 0.99)
        )
    )

    features = np.asarray(
        [
            *channel_means.tolist(),
            *channel_stds.tolist(),
            global_mean,
            global_std,
            _entropy(flattened),
            skewness,
            excess_kurtosis,
            dynamic_range,
            contrast_ratio,
            saturation_fraction,
        ],
        dtype=np.float64,
    )

    if features.shape != (14,):
        raise RuntimeError(
            f"Pixel feature dimension mismatch: {features.shape}"
        )

    return features


def _high_frequency_energy(
    grayscale: np.ndarray,
) -> float:
    spectrum = np.fft.fftshift(
        np.fft.fft2(grayscale)
    )

    power = np.abs(spectrum) ** 2

    height, width = grayscale.shape
    yy, xx = np.ogrid[:height, :width]

    center_y = (height - 1) / 2.0
    center_x = (width - 1) / 2.0

    radius = np.sqrt(
        (yy - center_y) ** 2
        + (xx - center_x) ** 2
    )

    maximum_radius = float(radius.max())

    high_frequency_mask = radius >= 0.25 * maximum_radius

    total_energy = float(power.sum())

    if total_energy <= EPSILON:
        return 0.0

    return float(
        power[high_frequency_mask].sum()
        / total_energy
    )


def _tamura_coarseness_approx(
    grayscale: np.ndarray,
) -> float:
    scales = (1, 2, 4, 8)
    responses: list[float] = []

    for scale in scales:
        kernel_size = 2 * scale + 1

        blurred = cv2.blur(
            grayscale,
            (kernel_size, kernel_size),
        )

        horizontal = np.mean(
            np.abs(
                blurred[:, 2 * scale:]
                - blurred[:, :-2 * scale]
            )
        )

        vertical = np.mean(
            np.abs(
                blurred[2 * scale:, :]
                - blurred[:-2 * scale, :]
            )
        )

        responses.append(float(horizontal + vertical))

    selected_scale = scales[int(np.argmax(responses))]

    # Normalize to [0, 1] for the configured scale range.
    return float(selected_scale / max(scales))


def _tamura_contrast(
    grayscale: np.ndarray,
) -> float:
    values = grayscale.astype(np.float64).reshape(-1)

    mean = values.mean()
    centered = values - mean

    variance = float(np.mean(centered ** 2))

    if variance <= EPSILON:
        return 0.0

    standard_deviation = math.sqrt(variance)

    fourth_moment = float(
        np.mean(centered ** 4)
    )

    kurtosis = fourth_moment / (
        variance ** 2 + EPSILON
    )

    return float(
        standard_deviation
        / (kurtosis ** 0.25 + EPSILON)
    )


def _tamura_directionality(
    gradient_x: np.ndarray,
    gradient_y: np.ndarray,
) -> float:
    magnitude = np.sqrt(
        gradient_x ** 2 + gradient_y ** 2
    )

    angles = (
        np.arctan2(gradient_y, gradient_x)
        + np.pi
    ) % np.pi

    mask = magnitude > 0.05

    if not np.any(mask):
        return 0.0

    histogram, _ = np.histogram(
        angles[mask],
        bins=16,
        range=(0.0, np.pi),
        weights=magnitude[mask],
    )

    probabilities = histogram.astype(np.float64)
    total = probabilities.sum()

    if total <= EPSILON:
        return 0.0

    probabilities /= total
    probabilities = probabilities[probabilities > 0]

    normalized_entropy = (
        -np.sum(
            probabilities
            * np.log(probabilities)
        )
        / np.log(16.0)
    )

    # Higher value means a more concentrated orientation distribution.
    return float(1.0 - normalized_entropy)


def _lbp_uniformity(
    grayscale: np.ndarray,
) -> float:
    grayscale_uint8 = np.clip(
        grayscale * 255.0,
        0,
        255,
    ).astype(np.uint8)

    points = 8

    lbp = local_binary_pattern(
        grayscale_uint8,
        P=points,
        R=1,
        method="uniform",
    )

    histogram, _ = np.histogram(
        lbp,
        bins=np.arange(0, points + 3),
        range=(0, points + 2),
    )

    probabilities = histogram.astype(np.float64)

    if probabilities.sum() <= 0:
        return 0.0

    probabilities /= probabilities.sum()

    return float(np.sum(probabilities ** 2))


def _single_image_texture_features(
    image_chw: np.ndarray,
) -> np.ndarray:
    image_hwc = np.transpose(image_chw, (1, 2, 0))

    grayscale = cv2.cvtColor(
        image_hwc.astype(np.float32),
        cv2.COLOR_RGB2GRAY,
    )

    gradient_x = cv2.Sobel(
        grayscale,
        cv2.CV_32F,
        1,
        0,
        ksize=3,
    )

    gradient_y = cv2.Sobel(
        grayscale,
        cv2.CV_32F,
        0,
        1,
        ksize=3,
    )

    gradient_magnitude = np.sqrt(
        gradient_x ** 2 + gradient_y ** 2
    )

    laplacian = cv2.Laplacian(
        grayscale,
        cv2.CV_32F,
        ksize=3,
    )

    values: list[float] = [
        float(gradient_magnitude.mean()),
        float(gradient_magnitude.std()),
        float(np.percentile(gradient_magnitude, 90)),
        float(np.mean(np.abs(gradient_x))),
        float(np.mean(np.abs(gradient_y))),
        float(np.mean(gradient_magnitude > 0.10)),
        float(np.mean(np.abs(laplacian))),
        float(np.std(laplacian)),
        float(np.var(laplacian)),
        _high_frequency_energy(grayscale),
    ]

    for angle_degrees in (0, 45, 90, 135):
        kernel = cv2.getGaborKernel(
            ksize=(9, 9),
            sigma=2.0,
            theta=np.deg2rad(angle_degrees),
            lambd=4.0,
            gamma=0.5,
            psi=0.0,
            ktype=cv2.CV_32F,
        )

        response = cv2.filter2D(
            grayscale,
            cv2.CV_32F,
            kernel,
        )

        absolute_response = np.abs(response)

        values.extend(
            [
                float(absolute_response.mean()),
                float(absolute_response.std()),
            ]
        )

    values.extend(
        [
            _tamura_coarseness_approx(grayscale),
            _tamura_contrast(grayscale),
            _tamura_directionality(
                gradient_x,
                gradient_y,
            ),
            _lbp_uniformity(grayscale),
        ]
    )

    features = np.asarray(values, dtype=np.float64)

    if features.shape != (22,):
        raise RuntimeError(
            f"Texture feature dimension mismatch: "
            f"{features.shape}"
        )

    return features


def extract_texture_features(
    unit_images: np.ndarray,
) -> np.ndarray:
    per_image_features = np.stack(
        [
            _single_image_texture_features(image)
            for image in unit_images
        ],
        axis=0,
    )

    return per_image_features.mean(axis=0)


@torch.inference_mode()
def extract_frozen_embeddings(
    unit_images: torch.Tensor,
    encoder: FrozenDescriptorEncoder,
    device: torch.device,
    batch_size: int = 128,
) -> np.ndarray:
    encoder.eval()

    embedding_batches: list[np.ndarray] = []

    for start in range(0, len(unit_images), batch_size):
        end = start + batch_size

        image_batch = unit_images[start:end].to(
            device,
            non_blocking=True,
        )

        embeddings = encoder(image_batch)

        embedding_batches.append(
            embeddings.detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )

    return np.concatenate(embedding_batches, axis=0)


def _pairwise_distances(
    values: np.ndarray,
) -> np.ndarray:
    differences = (
        values[:, None, :]
        - values[None, :, :]
    )

    return np.sqrt(
        np.maximum(
            np.sum(differences ** 2, axis=-1),
            0.0,
        )
    )


def extract_geometry_features(
    embeddings: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    unique_labels = np.unique(labels)

    prototypes: list[np.ndarray] = []
    intra_class_variances: list[float] = []

    for label in unique_labels:
        class_embeddings = embeddings[labels == label]

        if len(class_embeddings) == 0:
            raise RuntimeError(
                f"No embeddings found for label {label}"
            )

        prototype = class_embeddings.mean(axis=0)

        squared_distances = np.sum(
            (class_embeddings - prototype) ** 2,
            axis=1,
        )

        prototypes.append(prototype)
        intra_class_variances.append(
            float(np.mean(squared_distances))
        )

    prototypes_array = np.stack(prototypes, axis=0)
    intra_array = np.asarray(
        intra_class_variances,
        dtype=np.float64,
    )

    prototype_distances = _pairwise_distances(
        prototypes_array
    )

    upper_triangle = prototype_distances[
        np.triu_indices(
            len(prototypes_array),
            k=1,
        )
    ]

    if len(upper_triangle) == 0:
        raise RuntimeError(
            "At least two classes are required"
        )

    nearest_matrix = prototype_distances.copy()
    np.fill_diagonal(nearest_matrix, np.inf)

    nearest_distances = nearest_matrix.min(axis=1)

    prototype_norms = np.linalg.norm(
        prototypes_array,
        axis=1,
    )

    support_norms = np.linalg.norm(
        embeddings,
        axis=1,
    )

    global_centroid = embeddings.mean(axis=0)

    between_class_scatter = float(
        np.mean(
            np.sum(
                (
                    prototypes_array
                    - global_centroid
                ) ** 2,
                axis=1,
            )
        )
    )

    within_class_scatter = float(
        intra_array.mean()
    )

    fisher_ratio = between_class_scatter / (
        within_class_scatter + EPSILON
    )

    cluster_compactness = float(
        np.mean(np.sqrt(np.maximum(intra_array, 0.0)))
    )

    centroid_separation = float(
        upper_triangle.mean()
    )

    separation_to_variance = (
        centroid_separation
        / (
            math.sqrt(
                max(within_class_scatter, 0.0)
            )
            + EPSILON
        )
    )

    features = np.asarray(
        [
            float(intra_array.mean()),
            float(intra_array.std()),
            float(intra_array.min()),
            float(intra_array.max()),
            float(upper_triangle.mean()),
            float(upper_triangle.std()),
            float(upper_triangle.min()),
            float(upper_triangle.max()),
            float(nearest_distances.mean()),
            float(nearest_distances.std()),
            float(prototype_norms.mean()),
            float(prototype_norms.std()),
            float(support_norms.mean()),
            float(support_norms.std()),
            float(fisher_ratio),
            cluster_compactness,
            centroid_separation,
            float(separation_to_variance),
        ],
        dtype=np.float64,
    )

    if features.shape != (18,):
        raise RuntimeError(
            f"Geometry feature dimension mismatch: "
            f"{features.shape}"
        )

    return features


def extract_spectral_features(
    embeddings: np.ndarray,
) -> np.ndarray:
    centered = embeddings - embeddings.mean(
        axis=0,
        keepdims=True,
    )

    singular_values = np.linalg.svd(
        centered,
        full_matrices=False,
        compute_uv=False,
    )

    eigenvalues = singular_values ** 2

    total_variance = float(eigenvalues.sum())

    if total_variance <= EPSILON:
        return np.zeros(10, dtype=np.float64)

    explained = eigenvalues / total_variance

    padded = np.pad(
        explained,
        pad_width=(
            0,
            max(0, 10 - len(explained)),
        ),
        mode="constant",
    )

    positive = eigenvalues[eigenvalues > EPSILON]
    probabilities = positive / positive.sum()

    entropy = float(
        -np.sum(
            probabilities
            * np.log(probabilities)
        )
    )

    normalized_entropy = (
        entropy / np.log(len(probabilities))
        if len(probabilities) > 1
        else 0.0
    )

    effective_rank = float(np.exp(entropy))

    spectral_flatness = float(
        np.exp(
            np.mean(
                np.log(positive + EPSILON)
            )
        )
        / (
            np.mean(positive)
            + EPSILON
        )
    )

    anisotropy = float(
        positive.max()
        / (
            positive.mean()
            + EPSILON
        )
    )

    log_condition_number = float(
        np.log(
            (
                positive.max()
                + EPSILON
            )
            / (
                positive.min()
                + EPSILON
            )
        )
    )

    features = np.asarray(
        [
            float(padded[0]),
            float(padded[1]),
            float(padded[2]),
            float(padded[:5].sum()),
            float(padded[:10].sum()),
            effective_rank,
            normalized_entropy,
            spectral_flatness,
            anisotropy,
            log_condition_number,
        ],
        dtype=np.float64,
    )

    if features.shape != (10,):
        raise RuntimeError(
            f"Spectral feature dimension mismatch: "
            f"{features.shape}"
        )

    return features


def extract_zero_descriptor(
    *,
    support_images: torch.Tensor,
    support_labels: torch.Tensor,
    dataset_name: str,
    encoder: FrozenDescriptorEncoder,
    device: torch.device,
) -> np.ndarray:
    """
    Extract one 64-dimensional descriptor from one support set.
    """
    unit_images = denormalize_to_unit_interval(
        support_images.detach().cpu(),
        dataset_name=dataset_name,
    ).float()

    unit_images_numpy = unit_images.numpy()

    labels = (
        support_labels.detach()
        .cpu()
        .numpy()
        .astype(np.int64)
    )

    embeddings = extract_frozen_embeddings(
        unit_images=unit_images,
        encoder=encoder,
        device=device,
    )

    descriptor = np.concatenate(
        [
            extract_pixel_features(unit_images_numpy),
            extract_texture_features(unit_images_numpy),
            extract_geometry_features(
                embeddings,
                labels,
            ),
            extract_spectral_features(embeddings),
        ],
        axis=0,
    )

    if descriptor.shape != (64,):
        raise RuntimeError(
            f"Zero descriptor has invalid shape: "
            f"{descriptor.shape}"
        )

    if not np.all(np.isfinite(descriptor)):
        invalid_indices = np.flatnonzero(
            ~np.isfinite(descriptor)
        )

        invalid_names = [
            ZERO_FEATURE_NAMES[index]
            for index in invalid_indices
        ]

        raise FloatingPointError(
            "Non-finite descriptor values detected: "
            f"{invalid_names}"
        )

    return descriptor
