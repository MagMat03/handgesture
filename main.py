from src.data_loader import load_dataset
#from src.data_loader import check_dataset
#check_dataset("data")

samples = load_dataset("data")
print("Liczba próbek:", len(samples))
sample = samples[0]
print("Osoba:", sample["person"])
print("Tło:", sample["background"])
print("Gest:", sample["gesture"])
print("Plik:", sample["filename"])
print("Dłoń:", sample["handedness"])

print("Screen landmarks:",sample["screen_landmarks"].shape)
print("World:",sample["world_landmarks"].shape)
print("P0 world:",sample["world_landmarks"][0])
print("P5 world:",sample["world_landmarks"][5])
print("P17 world:",sample["world_landmarks"][17])

