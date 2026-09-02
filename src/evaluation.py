from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

def evaluate_model(y_test, predictions, model_name, results_dir):

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    accuracy = accuracy_score(y_test, predictions)

    macro_f1 = f1_score(y_test, predictions, average="macro")

    report = classification_report(y_test, predictions)

    print(f"\n{model_name}")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Macro F1: {macro_f1 * 100:.2f}%")

    print("\nClassification report:")
    print(report)

    safe_name = (model_name.lower().replace(" ", "_"))

    report_path = results_dir / f"{safe_name}_report.txt"

    with open(report_path,"w",encoding="utf-8") as file:

        file.write(f"Model: {model_name}\n")
        file.write(f"Accuracy: {accuracy * 100:.2f}%\n")
        file.write(f"Macro F1: {macro_f1 * 100:.2f}%\n\n")

        file.write("Classification report:\n")
        file.write(report)

    labels = sorted(set(y_test))

    matrix = confusion_matrix(y_test, predictions, labels=labels)

    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=labels)

    fig, ax = plt.subplots(figsize=(12, 10))

    display.plot(ax=ax, xticks_rotation=45, values_format="d",cmap="Blues")

    ax.set_title(f"Confusion Matrix - {model_name}")

    fig.tight_layout()

    matrix_path = (results_dir/ f"{safe_name}_confusion_matrix.png")

    fig.savefig(matrix_path, dpi=200, bbox_inches="tight")

    plt.close(fig)

    return accuracy, macro_f1

