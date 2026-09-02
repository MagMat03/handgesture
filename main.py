from config import (
    DATA_ROOT,
    RESULTS_DIR,
    TEST_PERSON,
    FEATURE_MODE
)

from src.data_loader import load_dataset
from src.features import build_feature_dataset
from src.training import (
    split_by_person,
    train_models
)

from src.evaluation import evaluate_model

samples = load_dataset(DATA_ROOT)
X, y, persons = build_feature_dataset(
    samples,
    mode=FEATURE_MODE
)

print("X:", X.shape)
print("y:", y.shape)
print("persons:", persons.shape)

X_train, X_test, y_train, y_test = split_by_person(
    X,
    y,
    persons,
    test_person=TEST_PERSON
)

rf, svm, mlp = train_models(
    X_train,
    y_train
)

rf_predictions = rf.predict(X_test)
svm_predictions = svm.predict(X_test)
mlp_predictions = mlp.predict(X_test)

experiment_dir = (
    RESULTS_DIR
    / TEST_PERSON
    / FEATURE_MODE
)


evaluate_model(
    y_test,
    rf_predictions,
    "Random Forest",
    experiment_dir
)

evaluate_model(
    y_test,
    svm_predictions,
    "SVM",
    experiment_dir
)

evaluate_model(
    y_test,
    mlp_predictions,
    "MLP",
    experiment_dir
)
























# #from src.data_loader import check_dataset
# #check_dataset("data")
# from src.features import (
#     signed_tetrahedron_volume,
#     normalize_volume,
#     extract_fingertip_volumes,
#     build_feature_dataset,
#     extract_all_volumes
# )
# from src.data_loader import (
#     load_dataset,
#     count_samples_by_gesture,
#     count_samples_by_person_and_gesture
# )
#
# from src.training import (
#     split_by_person,
#     train_models
# )
#
# from src.evaluation import evaluate_model
# from sklearn.metrics import accuracy_score
#
# samples = load_dataset("data")
# print("Liczba próbek:", len(samples))
# sample = samples[0]
# print("Osoba:", sample["person"])
# print("Tło:", sample["background"])
# print("Gest:", sample["gesture"])
# print("Plik:", sample["filename"])
# print("Dłoń:", sample["handedness"])
#
# print("Screen landmarks:",sample["screen_landmarks"].shape)
# print("World:",sample["world_landmarks"].shape)
# print("P0 world:",sample["world_landmarks"][0])
# print("P5 world:",sample["world_landmarks"][5])
# print("P17 world:",sample["world_landmarks"][17])
#
# gesture_counts = count_samples_by_gesture(samples)
# print("\nLiczba próbek dla każdego gestu:")
# for gesture, count in sorted(gesture_counts.items()):
#     print(f"{gesture}: {count}")
# counts = count_samples_by_person_and_gesture(samples)
# # for person in sorted(counts):
# #     print(f"\n{person}")
# #     for gesture, count in sorted(counts[person].items()):
# #         print(f"{gesture}: {count}")
#
# sample = samples[0]
# landmarks = sample["world_landmarks"]
# p0 = landmarks[0]
# p5 = landmarks[5]
# p17 = landmarks[17]
# p8 = landmarks[8]
# v8 = signed_tetrahedron_volume(p0, p5,p17, p8)
# # print("P0:", p0)
# # print("P5:", p5)
# # print("P17:", p17)
# # print("P8:", p8)
# # print("Objętość V8:", v8)
# v8 = signed_tetrahedron_volume(p0, p5, p17,p8)
# v8_normalized = normalize_volume(v8, p5,p17)
# # print("Surowe V8:", v8)
# # print("Znormalizowane V8:", v8_normalized)
#
# sample = samples[0]
#
# features = extract_fingertip_volumes(
#     sample["world_landmarks"]
# )
#
# # print("Gest:", sample["gesture"])
# # print("Cechy:", features)
# # print("Kształt:", features.shape)
#
# samples = load_dataset("data")
#
# X, y, persons = build_feature_dataset(samples)
#
# # print("X:", X.shape)
# # print("y:", y.shape)
# # print("persons:", persons.shape)
# #
# # print("Pierwszy wektor cech:", X[0])
# # print("Pierwszy gest:", y[0])
# # print("Pierwsza osoba:", persons[0])
# #
# # features = extract_all_volumes(
# #     sample["world_landmarks"]
# # )
#
# # print(features)
# # print(features.shape)
#
#
# X5, y5, persons5 = build_feature_dataset(
#     samples,
#     mode="fingertips"
# )
#
# X18, y18, persons18 = build_feature_dataset(
#     samples,
#     mode="all"
# )
#
# print(X5.shape)
# print(X18.shape)
#
#
#
# X_train, X_test, y_train, y_test = split_by_person(
#     X5,
#     y5,
#     persons5,
#     test_person="P05"
# )
#
# print("X_train:", X_train.shape)
# print("X_test:", X_test.shape)
#
# print("y_train:", y_train.shape)
# print("y_test:", y_test.shape)
#
#
# X_train, X_test, y_train, y_test = split_by_person(
#     X18,
#     y18,
#     persons18,
#     test_person="P05"
# )
#
# rf, svm, mlp = train_models(
#     X_train,
#     y_train
# )
#
# # rf_predictions = rf.predict(X_test)
# # svm_predictions = svm.predict(X_test)
# # mlp_predictions = mlp.predict(X_test)
# #
# # print("Prawdziwe:", y_test[:20])
# # print("RF:       ", rf_predictions[:20])
# # print("SVM:      ", svm_predictions[:20])
# # print("MLP:      ", mlp_predictions[:20])
# #
# # rf_accuracy = accuracy_score(y_test, rf_predictions)
# # svm_accuracy = accuracy_score(y_test, svm_predictions)
# # mlp_accuracy = accuracy_score(y_test, mlp_predictions)
# #
# # print(f"RF accuracy:  {rf_accuracy * 100:.2f}%")
# # print(f"SVM accuracy: {svm_accuracy * 100:.2f}%")
# # print(f"MLP accuracy: {mlp_accuracy * 100:.2f}%")
#
# rf_predictions = rf.predict(X_test)
# svm_predictions = svm.predict(X_test)
# mlp_predictions = mlp.predict(X_test)
# evaluate_model(
#     y_test,
#     rf_predictions,
#     "Random Forest"
# )
#
# evaluate_model(
#     y_test,
#     svm_predictions,
#     "SVM"
# )
#
# evaluate_model(
#     y_test,
#     mlp_predictions,
#     "MLP"
# )