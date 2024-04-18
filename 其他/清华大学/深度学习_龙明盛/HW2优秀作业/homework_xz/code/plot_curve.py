import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import click
import os
from pathlib import Path


@click.command()
@click.option('-i', '--log-file', type=str, required=True)
def main(log_file: str):
    df = pd.read_csv(log_file)
    os.makedirs('figures', exist_ok=True)
    
    sns.set()
    plt.plot(df['epoch'], df['train_loss'], label='Train')
    plt.plot(df['epoch'], df['valid_loss'], label='Valid')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig(os.path.join('figures', Path(log_file).name.replace('.csv', '_loss.pdf')))
    plt.cla()

    plt.plot(df['epoch'], df['train_acc'], label='Train')
    plt.plot(df['epoch'], df['valid_acc'], label='Valid')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.savefig(os.path.join('figures', Path(log_file).name.replace('.csv', '_acc.pdf')))
    plt.cla()


if __name__ == '__main__':
    main()
