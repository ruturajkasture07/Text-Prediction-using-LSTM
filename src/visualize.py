import json
import os
import sys

# Try to import matplotlib, give a helpful error if missing
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend to avoid display issues
    import matplotlib.pyplot as plt
except ImportError:
    print("matplotlib is not installed. Run: pip install matplotlib")
    sys.exit(1)

def plot_history(history_path='models/history.json', output_path='models/loss_curve.png'):
    if not os.path.exists(history_path):
        print(f"History file not found at '{history_path}'.")
        print("The model was trained before history tracking was added.")
        print("Please re-train the model with the updated train.py to generate history.")
        print("  Command: python src/train.py")
        return

    with open(history_path, 'r') as f:
        history = json.load(f)

    epochs = range(1, len(history['train_loss']) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['train_loss'], label='Train Loss', color='#6366f1', linewidth=2, marker='o', markersize=4)
    plt.plot(epochs, history['val_loss'], label='Validation Loss', color='#c084fc', linewidth=2, marker='s', markersize=4)
    plt.title('Model Training History', fontsize=14, fontweight='bold')
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss (Cross-Entropy)', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    plt.savefig(output_path, dpi=150)
    print(f"Loss curve saved to {output_path}")
    print("Open the image file to view the training/validation loss curves.")

if __name__ == "__main__":
    plot_history()
