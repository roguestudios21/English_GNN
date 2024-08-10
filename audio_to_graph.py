import os
import librosa
import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Data
from pathlib import Path


def extract_features(audio_path, sr=22050, n_mfcc=13, n_mels=128):
    """
    Extracts Mel spectrogram and MFCC features from an audio file.

    Parameters:
    - audio_path (str): Path to the audio file (supports .wav, .mp3, etc.)
    - sr (int): Sampling rate (default: 22050)
    - n_mfcc (int): Number of MFCC features to extract (default: 13)
    - n_mels (int): Number of Mel bands to generate (default: 128)

    Returns:
    - mel_spectrogram_db (ndarray): Mel spectrogram in dB
    - mfcc (ndarray): MFCC features
    """
    try:
        # Load the audio file
        y, sr = librosa.load(audio_path, sr=sr)

        # Extract Mel spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

        # Extract MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

        return mel_spectrogram_db, mfcc

    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None, None


def construct_graph(mel_spectrogram, mfcc):
    """
    Constructs a graph from Mel spectrogram and MFCC features.

    Parameters:
    - mel_spectrogram (ndarray): Mel spectrogram features
    - mfcc (ndarray): MFCC features

    Returns:
    - G (networkx.Graph): Graph with nodes as time frames and features as node attributes
    """
    num_frames = mel_spectrogram.shape[1]

    # Combine features
    combined_features = np.concatenate((mel_spectrogram, mfcc), axis=0)

    # Create a graph
    G = nx.Graph()

    for i in range(num_frames):
        G.add_node(i, feature=combined_features[:, i])

    # Add edges (you can customize the connectivity)
    for i in range(num_frames - 1):
        G.add_edge(i, i + 1)

    return G


def graph_to_torch_data(graph):
    """
    Converts a NetworkX graph to a PyTorch Geometric Data object.

    Parameters:
    - graph (networkx.Graph): Input graph

    Returns:
    - data (torch_geometric.data.Data): PyTorch Geometric Data object
    """
    # Convert NetworkX graph to PyTorch Geometric Data
    x = torch.tensor([graph.nodes[i]['feature'] for i in range(graph.number_of_nodes())], dtype=torch.float)
    edge_index = torch.tensor(list(graph.edges), dtype=torch.long).t().contiguous()

    data = Data(x=x, edge_index=edge_index)
    return data


# Example usage
input_path = ''  # Update this path to your input directory
output_path = ''

# Ensure the output directory exists
os.makedirs(output_path, exist_ok=True)

# Process all audio files in the input directory
audio_files = list(Path(input_path).rglob('*.mp3')) + list(Path(input_path).rglob('*.wav'))

for audio_file in audio_files:
    mel_spectrogram, mfcc = extract_features(audio_file)

    if mel_spectrogram is not None and mfcc is not None:
        graph = construct_graph(mel_spectrogram, mfcc)
        torch_data = graph_to_torch_data(graph)

        # Save the processed graph data to the output directory
        output_file_path = os.path.join(output_path, f'{audio_file.stem}_graph.pt')
        torch.save(torch_data, output_file_path)

        print(f'Processed {audio_file.name}: {torch_data}')
    else:
        print(f'Skipped {audio_file.name} due to an error.')