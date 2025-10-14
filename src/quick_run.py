import os
import subprocess
import sys


def model_loop():
    args = ["--model"]
    while True:
        print("Choose model")
        print("\t1. CNN")
        print("\t2. ResNet50")
        choice = input(">> ")
        match choice:
            case "1":
                args.append("CNN")
                return args
            case "2":
                args.append("ResNet50")
                return args
            case _:
                print("Invalid choice. try again")


def augmentation_loop():
    args = ["--aug"]
    while True:
        print("Choose augmentation type")
        print("\t1. Normal (Default)")
        print("\t2. Strong")
        print("\t3. No Augmentation")
        choice = input(">> ")
        match choice:
            case "2":
                args.append("strong_augment")
            case "3":
                args.append("no_augment")
            case _:
                args.append("normal_augment")
        return args


def seed_loop():
    args = ["--seed"]
    while True:
        print("Choose seed for random operations [1-9999] (Default: 7)")
        try:
            choice = input(">> ")
            if not choice:
                choice = "7"
            int_choice = int(choice)
            if int_choice < 1 or int_choice > 9999:
                raise ValueError(f"Invalid seed: {int_choice}")
            args.append(choice)
            return args
        except ValueError:
            print("Invalid seed")


def epochs_loop():
    args = ["--epochs"]
    while True:
        print("Choose max epochs for training [30-200] (Default: 100)")
        try:
            choice = input(">> ")
            if not choice:
                choice = "100"
            int_choice = int(choice)
            if int_choice < 30 or int_choice > 200:
                raise ValueError(f"Invalid epoch count: {int_choice}")
            args.append(choice)
            return args
        except ValueError:
            print("Invalid count")


def batch_loop():
    args = ["--batch"]
    while True:
        print("Choose batch size for training")
        print("\t1. 16")
        print("\t2. 32 (Default)")
        print("\t3. 64")
        print("\t4. 128")
        choice = input(">> ")
        match choice:
            case "1":
                args.append("16")
            case "3":
                args.append("64")
            case "4":
                args.append("128")
            case _:
                args.append("32")
        return args


def evaluate_loop():
    print("Are you doing evaluation only? (Model already trained)")
    print("\t1. Yes")
    print("\t2. No (Default)")
    choice = input(">> ")
    return True if choice == "1" else False


def train_loop():
    args = [sys.executable, "-m", "src.run"]
    args.extend(model_loop())
    args.extend(augmentation_loop())
    args.extend(seed_loop())
    args.extend(epochs_loop())
    args.extend(batch_loop())
    if evaluate_loop():
        args.append("--evaluate")
    subprocess.Popen(args, env=os.environ)


def view_results():
    args = [sys.executable, "-m", "src.ui.results"]
    subprocess.Popen(args, env=os.environ)


def main_loop():
    print("Galaxy Classification")
    while True:
        print("Options:")
        print("\t1. Train/Evaluate")
        print("\t2. View results")
        print("\tX. Exit")
        choice = input(">> ").upper()
        match choice:
            case "1":
                train_loop()
            case "2":
                view_results()
            case "X":
                exit()
            case _:
                print("Invalid choice")


if __name__ == '__main__':
    main_loop()
