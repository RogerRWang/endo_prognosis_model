from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix


def get_ppv_and_npv(y_test, y_pred, model: str):
    """
    Positive Predictive Value and Negative Predictive Value
    """
    # Compute confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    # Calculate PPV and NPV
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0  # Handle division by zero
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0

    # Data for plotting
    labels = ['PPV', 'NPV']
    values = [ppv, npv]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, values, color=['blue', 'green'])

    ax.set_ylim(0, 1)
    ax.set_ylabel('Score')
    ax.set_title(f'PPV & NPV ({model})')

    # Display the values on top of the bars
    for i, value in enumerate(values):
        ax.text(i, value + 0.02, f'{value:.3f}', ha='center', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'./output/visualizations/ppv_npv_{model}.png')

