import numpy as np

BASE_POINTS = [0, 5, 17]
FINGERTIPS = [4, 8, 12, 16, 20]
ALL_FEATURE_POINTS = [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 20]

def signed_tetrahedron_volume(p0, p5, p17, pi):

    vector_1 = p5 - p0
    vector_2 = p17 - p0
    vector_3 = pi - p0

    cross_product = np.cross(vector_1, vector_2)

    scalar_triple_product = np.dot(
        cross_product,
        vector_3
    )

    volume = scalar_triple_product / 6.0

    return volume

#normalizacja dłoni wgledem punktów P5 i P17 (szerokośc dłoni)
def normalize_volume(volume, p5, p17):

    scale = np.linalg.norm(p17 - p5)

    normalized_volume = volume / (scale ** 3)

    return normalized_volume

def extract_fingertip_volumes(landmarks):

    p0 = landmarks[0]
    p5 = landmarks[5]
    p17 = landmarks[17]
    features = []
    for index in FINGERTIPS:
        pi = landmarks[index]
        volume = signed_tetrahedron_volume(p0, p5, p17, pi)
        normalized_volume = normalize_volume(volume, p5,p17)
        features.append(normalized_volume)

    return np.array(features)


def extract_all_volumes(landmarks):

    p0 = landmarks[0]
    p5 = landmarks[5]
    p17 = landmarks[17]

    features = []

    for index in ALL_FEATURE_POINTS:

        pi = landmarks[index]

        volume = signed_tetrahedron_volume(p0, p5, p17, pi)
        normalized_volume = normalize_volume(volume, p5, p17)
        features.append(normalized_volume)

    for index, value in zip(
         ALL_FEATURE_POINTS,
         features
    ):
        print(f"P{index}: {value}")

    return np.array(features)


def build_feature_dataset(samples, mode="fingertips"):

    X = []
    y = []
    persons = []

    for sample in samples:

        if mode == "fingertips":
            features = extract_fingertip_volumes(sample["world_landmarks"])

        elif mode == "all":
            features = extract_all_volumes(sample["world_landmarks"])

        else:
            raise ValueError(f"Nieznany tryb cech: {mode}")

        X.append(features)
        y.append(sample["gesture"])
        persons.append(sample["person"])

    return (np.array(X), np.array(y), np.array(persons))
