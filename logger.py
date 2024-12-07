def print_model_config(num_params, batch_size):
    print("\nModel Configuration:")
    print("="*70)
    print(f"Total Parameters: {num_params:,}")
    print(f"Architecture: 8→16→32→32 channels")
    print(f"Batch Size: {batch_size}")
    print(f"Initial LR: 0.015")
    print(f"Max LR: 0.02")
    print(f"Weight Decay: 0.0001")
    print(f"Scheduler: OneCycleLR (pct_start=0.1, div_factor=10.0)")
    print(f"Dropout: Early 2% → Mid 5% → Late 10%")
    print("="*70 + "\n")

def print_training_strategy(batch_size, half_batch, steps_per_epoch):
    print(f"Training Strategy:")
    print("="*70)
    print(f"Training for 20 epochs with batch_size={batch_size}")
    print(f"Total steps per epoch: {steps_per_epoch}")
    print("="*70 + "\n")

def print_epoch_start(epoch, batch_size, lr, mode="Random batches"):
    print(f"\nEpoch {epoch + 1}")
    print("----------------------------------------")
    print(f"Batch Size: {batch_size}")
    print(f"Learning Rate: {lr:.6f}")
    print(f"Training mode: {mode}")

def print_batch_progress(batch_idx, total_batches, loss, accuracy, lr):
    print(f'Batch {batch_idx}/{total_batches}, '
          f'Loss: {loss:.4f}, '
          f'Accuracy: {accuracy:.2f}%')
    print(f'Learning rate: {lr:.6f}')

def print_epoch_summary(epoch, train_acc, test_acc, avg_loss):
    print(f'\nEpoch {epoch + 1} Summary:')
    print(f'Training Accuracy: {train_acc:.2f}%')
    print(f'Test Accuracy: {test_acc:.2f}%')
    print(f'Average Loss: {avg_loss:.4f}')

def print_training_summary(results, num_params, batch_size, best_accuracy):
    print("\n" + "="*100)
    print("Training Summary:")
    print("="*100)
    print(f"Model Parameters: {num_params:,}")
    print(f"Batch Size: {batch_size}")
    print(f"Initial LR: 0.015, Max LR: 0.02")
    print(f"Architecture: 8→16→32→32 channels")
    print("-"*100)
    print(f"{'Epoch':<10} {'Train Accuracy':<20} {'Test Accuracy':<20} {'Loss':<10} {'LR':<10}")
    print("-"*100)
    
    for epoch_results in results:
        print(f"{epoch_results['epoch']:<10} "
              f"{epoch_results['train_acc']:>18.2f}% "
              f"{epoch_results['test_acc']:>18.2f}% "
              f"{epoch_results['loss']:>10.4f} "
              f"{epoch_results['lr']:>10.6f}")
    
    print("="*100)
    print(f"Best Test Accuracy: {best_accuracy:.2f}%")
    print("="*100 + "\n")

def print_device_info(device):
    print(f"Using device: {device}")

def print_confidence_calculation_start():
    print("\nCalculating confidence scores for next epoch...")

def print_confidence_calculation_end():
    print("Sorting completed.")

def print_best_accuracy(accuracy):
    print(f'New best test accuracy: {accuracy:.2f}%')

def print_early_stopping(epoch):
    print(f'\nEarly stopping triggered after {epoch + 1} epochs') 