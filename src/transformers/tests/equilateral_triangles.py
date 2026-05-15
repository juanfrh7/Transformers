import numpy as np

def rotation_matrix(theta):
    """2D rotation matrix."""
    return np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])


def equilateral_score(centers):
    """
    Measures how close three points are to forming an equilateral triangle.
    Lower is more equilateral.
    """
    c1, c2, c3 = centers

    d12 = np.linalg.norm(c1 - c2)
    d13 = np.linalg.norm(c1 - c3)
    d23 = np.linalg.norm(c2 - c3)

    distances = np.array([d12, d13, d23])

    return (distances.max() - distances.min()) / distances.max()


def centers_inside_image(centers, image_size, margin):
    """Check whether all centers lie inside the image."""
    return np.all(
        (centers[:, 0] >= margin) &
        (centers[:, 0] < image_size - margin) &
        (centers[:, 1] >= margin) &
        (centers[:, 1] < image_size - margin)
    )


def sample_positive_centers(
    image_size=64,
    margin=8,
    min_side=12,
    max_side=36,
    rng=None
):
    """
    Generate three centers forming an equilateral triangle.
    """
    if rng is None:
        rng = np.random.default_rng()

    for _ in range(10_000):
        c1 = rng.uniform(margin, image_size - margin, size=2)

        side = rng.uniform(min_side, max_side)
        angle = rng.uniform(0, 2 * np.pi)

        direction = np.array([np.cos(angle), np.sin(angle)])
        c2 = c1 + side * direction

        sign = rng.choice([-1, 1])
        c3 = c1 + rotation_matrix(sign * np.pi / 3) @ (c2 - c1)

        centers = np.stack([c1, c2, c3], axis=0)

        if centers_inside_image(centers, image_size, margin):
            return centers

    raise RuntimeError("Could not sample valid positive centers.")


def sample_negative_centers(
    image_size=64,
    margin=8,
    min_distance=10,
    equilateral_threshold=0.18,
    rng=None
):
    """
    Generate three random centers that do not form an approximately equilateral triangle.
    """
    if rng is None:
        rng = np.random.default_rng()

    for _ in range(10_000):
        centers = rng.uniform(margin, image_size - margin, size=(3, 2))

        d12 = np.linalg.norm(centers[0] - centers[1])
        d13 = np.linalg.norm(centers[0] - centers[2])
        d23 = np.linalg.norm(centers[1] - centers[2])

        if min(d12, d13, d23) < min_distance:
            continue

        score = equilateral_score(centers)

        if score > equilateral_threshold:
            return centers

    raise RuntimeError("Could not sample valid negative centers.")


def draw_cluster_image(
    centers,
    image_size=64,
    points_per_cluster=20,
    cluster_std=1.4,
    point_radius=1,
    rng=None
):
    """
    Draw an image with three clusters of points.
    """
    if rng is None:
        rng = np.random.default_rng()

    image = np.zeros((image_size, image_size), dtype=np.float32)

    for center in centers:
        points = rng.normal(loc=center, scale=cluster_std, size=(points_per_cluster, 2))

        for x, y in points:
            px = int(round(x))
            py = int(round(y))

            for dx in range(-point_radius, point_radius + 1):
                for dy in range(-point_radius, point_radius + 1):
                    xx = px + dx
                    yy = py + dy

                    if 0 <= xx < image_size and 0 <= yy < image_size:
                        image[yy, xx] = 1.0

    return image


def generate_example(
    label,
    image_size=64,
    rng=None
):
    """
    Generate one example.

    label = 1 means equilateral.
    label = 0 means non-equilateral.
    """
    if rng is None:
        rng = np.random.default_rng()

    if label == 1:
        centers = sample_positive_centers(image_size=image_size, rng=rng)
    elif label == 0:
        centers = sample_negative_centers(image_size=image_size, rng=rng)
    else:
        raise ValueError("label must be 0 or 1")

    image = draw_cluster_image(centers, image_size=image_size, rng=rng)

    return image, label, centers


def generate_dataset(
    n_samples,
    image_size=64,
    seed=42
):
    """
    Generate a balanced dataset.

    Returns:
        X: images with shape (n_samples, image_size, image_size)
        y: labels with shape (n_samples,)
        centers: cluster centers with shape (n_samples, 3, 2)
    """
    rng = np.random.default_rng(seed)

    X = np.zeros((n_samples, image_size, image_size), dtype=np.float32)
    y = np.zeros((n_samples,), dtype=np.int64)
    centers_all = np.zeros((n_samples, 3, 2), dtype=np.float32)

    for idx in range(n_samples):
        label = idx % 2
        image, label, centers = generate_example(label, image_size=image_size, rng=rng)

        X[idx] = image
        y[idx] = label
        centers_all[idx] = centers

    # Shuffle dataset
    perm = rng.permutation(n_samples)

    return X[perm], y[perm], centers_all[perm]