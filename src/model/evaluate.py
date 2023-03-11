import torch

from src.data.consts.piece_consts import REVERSED_PIECE_TYPE, REVERSED_PIECE_COLOR, NON_PIECE, PIECE_COLOR


def eval_model(model, loader: torch.utils.data.DataLoader, state: str, verbose: bool = True):
    """Evaluates the per-class type and color accuracy, as well as a balanced accuracy for type and class

    Args:
        model: trained model
        loader: data loader with a dataset to test the performance of the model over
        state: the type of dataset the model is run on (train or test)
        verbose: prints metrics if True, silent otherwise

    """
    model.eval()
    with torch.no_grad():
        num_piece_classes = len(REVERSED_PIECE_TYPE)
        num_piece_colors = len(REVERSED_PIECE_COLOR)

        type_correct_hits = torch.zeros(num_piece_classes)
        type_counts = torch.zeros(num_piece_classes)
        color_correct_hits = torch.zeros(num_piece_colors)
        color_counts = torch.zeros(num_piece_colors)

        for i, (image, type_label, color_label, is_piece) in enumerate(loader):
            type_pred_probs, color_pred_probs = model(image)
            type_pred = torch.argmax(type_pred_probs, axis=1)
            color_pred = torch.argmax(color_pred_probs, axis=1)

            type_label = torch.argmax(type_label, axis=1)
            color_label = torch.argmax(color_label, axis=1)

            type_correct_hits[type_label] += (type_pred == type_label).to(torch.int64)
            labels_count_per_class = torch.bincount(type_label)
            type_counts[labels_count_per_class.nonzero()] += labels_count_per_class[labels_count_per_class.nonzero()]

            is_piece_idx = is_piece.nonzero()

            if len(is_piece_idx) > 0:
                color_correct_hits[color_label[is_piece_idx]] += \
                    (color_pred[is_piece_idx] == color_label[is_piece_idx]).to(torch.int64)
                color_count_per_class = torch.bincount(color_label)
                color_counts[color_count_per_class.nonzero()] += color_count_per_class[color_count_per_class.nonzero()]

        type_accuracy = torch.nan_to_num(type_correct_hits / type_counts)
        color_accuracy = torch.nan_to_num(color_correct_hits / color_counts)

        type_rates = type_counts / type_counts.sum()
        color_rates = color_counts / color_counts.sum()
        balanced_type_accuracy = type_accuracy @ type_rates
        balanced_color_accuracy = color_accuracy @ color_rates

        if verbose:
            print(f'\nResults over the {state} set\n')
            for piece_type, piece_name in REVERSED_PIECE_TYPE.items():
                print(f'Accuracy for {piece_name}: {type_accuracy[piece_type]:.3f}')

            for piece_color, color_name in REVERSED_PIECE_COLOR.items():
                print(f'Accuracy for {color_name}: {color_accuracy[piece_color]:.3f}')

            print()
            print(f'Balanced type accuracy: {balanced_type_accuracy.item():.3f}')
            print(f'Balanced color accuracy: {balanced_color_accuracy.item():.3f}')

        return balanced_type_accuracy, balanced_color_accuracy
