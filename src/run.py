import argparse
import os
import pickle
from pathlib import Path

import keras

from src.classifier.galaxy_classifier import GalaxyClassifier
from src.model.model_types import ModelType
from src.preprocessing.augmentation import Augmentation

keras.mixed_precision.set_global_policy("mixed_float16")
os.environ["TF_DETERMINISTIC_OPS"] = "0"
os.environ["TF_CUDNN_DETERMINISTIC"] = "0"


def save_results(results: dict, model_type: ModelType, augmentation: Augmentation, epochs: int, evaluation: bool):
    """Save results to a pickle file."""
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    # Create filename based on model, augmentation, and epochs
    filename = f"{model_type.value.lower()}_{augmentation.value}_{epochs}_epochs_{'evaluation_' if evaluation else ''}results.pkl"
    filepath = results_dir / filename

    try:
        with open(filepath, 'wb') as f:
            pickle.dump(results, f)
        print(f"\nResults saved to: {filepath}")
        return filepath
    except Exception as e:
        print(f"Error saving results: {e}")
        return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train or evaluate a model on the Galaxy10 dataset.")
    parser.add_argument("--model", type=str, choices=[model.value for model in ModelType],
                        help="Model to use for classification", required=True)
    parser.add_argument("--aug", type=str, choices=[aug.value for aug in Augmentation],
                        default="normal_augment",
                        help="What augmentation to use for training. (Default 'normal_augment')")
    parser.add_argument("--seed", type=int, default=7, help="Seed for random operations. (Default 7)")
    parser.add_argument("--batch", type=int, choices=[16, 32, 64, 128], default=32,
                        help="Batch size for training. (Default 32")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Max epoch count for training in range [30, 200]. (Default 50)")
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation on trained model")

    args = parser.parse_args()

    # Validate arguments
    try:
        model_type = ModelType(args.model)
    except ValueError:
        raise ValueError(f"Invalid model type {args.model}")

    try:
        augmentation = Augmentation(args.aug.lower())
    except ValueError:
        raise ValueError(f"Invalid augmentation {args.aug}")

    if args.epochs not in range(30, 201):
        raise ValueError(f"Invalid number of epochs {args.epochs}")

    seed = args.seed
    keras.utils.set_random_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    file_path = f"models/{model_type.value.lower()}_{augmentation.value}_{args.epochs}_epochs.keras"
    classifier = GalaxyClassifier(model_type, file_path)

    results = {
        'model_name': f"{model_type.value} ({augmentation.value})",
        'history': None,
        'y_true': None,
        'y_pred': None,
        'test_accuracy': None,
        'test_loss': None,
    }

    try:
        if not args.evaluate:
            print("=" * 60)
            print("TRAINING MODE")
            print("=" * 60)
            print(f"Model: {model_type.value}")
            print(f"Augmentation: {augmentation.value}")
            print(f"Epochs: {args.epochs}")
            print(f"Batch Size: {args.batch}")
            print(f"Seed: {seed}")
            print("=" * 60)

            train, val, test, class_weights = classifier.prepare_dataset(augmentation, args.batch, seed)
            history_dict = classifier.train(train, val, class_weights, args.epochs, plot_history=False)
            test_acc, test_loss, y_true, y_pred = classifier.evaluate(test, args.batch, plot_matrix=False)

            results['history'] = history_dict
            results['y_true'] = y_true
            results['y_pred'] = y_pred
            results['test_accuracy'] = test_acc
            results['test_loss'] = test_loss

            print(f"\nTraining Results:")
            print(f"  Test Accuracy: {test_acc:.4f} ({test_acc * 100:.2f}%)")
            print(f"  Test Loss: {test_loss:.4f}")

        else:
            print("=" * 60)
            print("EVALUATION MODE")
            print("=" * 60)
            print(f"Model: {model_type.value}")
            print(f"Augmentation: {augmentation.value}")
            print(f"Batch Size: {args.batch}")
            print(f"Seed: {seed}")
            print("=" * 60)

            _, _, test, _ = classifier.prepare_dataset(augmentation, args.batch, seed)
            test_acc, test_loss, y_true, y_pred = classifier.evaluate(test, args.batch, plot_matrix=False)

            results['y_true'] = y_true
            results['y_pred'] = y_pred
            results['test_accuracy'] = test_acc
            results['test_loss'] = test_loss

            print(f"\nEvaluation Results:")
            print(f"  Test Accuracy: {test_acc:.4f} ({test_acc * 100:.2f}%)")
            print(f"  Test Loss: {test_loss:.4f}")

        save_results(results, model_type, augmentation, args.epochs, args.evaluate)

    except Exception as e:
        print(f"\nError during training/evaluation: {e}")
        raise
    finally:
        # Cleanup
        try:
            keras.backend.clear_session(True)
        except Exception:
            pass
