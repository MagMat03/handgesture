from pathlib import Path
import re
import numpy as np

NUM_LANDMARKS=21
NUMBER = r"-?\d+(?:\.\d+)?(?:e-?\d+)?" #[opcjonalny minus][cyfry][opcjonalnie .cyfry][opcjonalnie e-minus-cyfry]

def _parse_landmarks(section:str, landmark_type:str):
    pattern = (
        rf"{landmark_type}\("
        rf"x=({NUMBER}), "
        rf"y=({NUMBER}), "
        rf"z=({NUMBER})"
    )
    matches=re.findall(pattern,section)
    landmarks=[]

    for x,y,z in matches:
        landmarks.append([
            float(x),
            float(y),
            float(z)
        ])

    landmarks=np.array(landmarks)

    if len(landmarks) != NUM_LANDMARKS:
        raise ValueError(
            f"Oczekiwano 21 landmarków, "
            f"znaleziono{(len(landmarks))}"
        )

    return landmarks

def load_sample(file_path: str | Path):
    text=file_path.read_text(encoding="utf-8")

    # Punky znormalizowane wzgledem obrazu "NormalizedLandmarks"
    normalized_match=re.search(
    r"hand_landmarks=\[\[(.*?)\]\],\s*"
    r"hand_world_landmarks=",
    text,
    flags=re.DOTALL)

    if normalized_match is None:
        raise ValueError("Nie znaleziono hand_landmarks.")

    screen_landmarks=_parse_landmarks(normalized_match.group(1),"NormalizedLandmark")


    # Punky znormalizowane wzgledem nadgarstaka "Landmark"
    world_match = re.search(
        r"hand_world_landmarks=\[\[(.*?)\]\]\)",
        text,
        flags=re.DOTALL
    )

    if world_match is None:
        raise ValueError("Nie znaleziono hand_world_landmarks.")

    world_landmarks = _parse_landmarks(world_match.group(1), "Landmark")

    handedness_match=re.search(
        r"category_name='([^']+)'",
        text
    )
    if handedness_match:
        handedness = handedness_match.group(1)
    else:
        handedness = None

    return {
        "screen_landmarks": screen_landmarks,
        "world_landmarks": world_landmarks,
        "handedness": handedness
    }


def load_dataset(data_root: str|Path):

    data_root=Path(data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"Folder danych nie istnieje: {data_root}")

    txt_files=sorted(data_root.glob("*/*/*/data/*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"Nie znaleziono plików .txt w {data_root}")

    samples=[]
    for file_path in txt_files:
        relative_path=file_path.relative_to(data_root)
        person=relative_path.parts[0]
        background = relative_path.parts[1]
        gesture = relative_path.parts[2]

        try:
            parsed=load_sample(file_path)
        #tymczasowo
        except Exception as error:
            print(
                f"Pominięto plik {file_path}: {error}"
            )
            continue
        #except Exception as error:
            #raise ValueError(f"Błąd pliku {file_path}: {error}") from error

        samples.append(
            {
                "person": person,
                "background": background,
                "gesture": gesture,
                "filename": file_path.name,
                "screen_landmarks": parsed["screen_landmarks"],
                "world_landmarks": parsed["world_landmarks"],
                "handedness": parsed["handedness"]
            }
        )
    return samples




###funkcja kontrolna ile jest zepsutych plików:
def check_dataset(data_root):
    data_root = Path(data_root)

    txt_files = sorted(
        data_root.glob("*/*/*/data/*.txt")
    )

    correct_files = []
    incorrect_files = []

    for file_path in txt_files:

        text = file_path.read_text(
            encoding="utf-8"
        )

        normalized_match = re.search(
            r"hand_landmarks=\[\[(.*?)\]\],\s*"
            r"hand_world_landmarks=",
            text,
            flags=re.DOTALL
        )

        if normalized_match is None:
            incorrect_files.append(
                (file_path, "Brak hand_landmarks")
            )
            continue

        section = normalized_match.group(1)

        pattern = (
            rf"NormalizedLandmark\("
            rf"x=({NUMBER}),\s*"
            rf"y=({NUMBER}),\s*"
            rf"z=({NUMBER})"
        )

        matches = re.findall(
            pattern,
            section
        )

        number_of_landmarks = len(matches)

        if number_of_landmarks == 21:

            correct_files.append(file_path)

        else:

            incorrect_files.append(
                (
                    file_path,
                    number_of_landmarks
                )
            )

    print("Wszystkich plików:", len(txt_files))
    print("Poprawnych:", len(correct_files))
    print("Niepoprawnych:", len(incorrect_files))

    print("\nNiepoprawne pliki:")

    for file_path, problem in incorrect_files:
        print(
            file_path,
            "->",
            problem
        )


def count_samples_by_gesture(samples):

    gesture_counts = {}

    for sample in samples:
        gesture = sample["gesture"]

        if gesture not in gesture_counts:
            gesture_counts[gesture] = 0

        gesture_counts[gesture] += 1

    return gesture_counts


def count_samples_by_person_and_gesture(samples):

    counts = {}

    for sample in samples:

        person = sample["person"]
        gesture = sample["gesture"]

        if person not in counts:
            counts[person] = {}

        if gesture not in counts[person]:
            counts[person][gesture] = 0

        counts[person][gesture] += 1

    return counts