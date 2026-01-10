"""
Data loading utilities for novels and training/test data.
"""
import csv
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

from config import DATASET_DIR


@dataclass
class Sample:
    """Represents a single backstory sample."""
    id: int
    book_name: str
    char: str
    caption: str
    content: str
    label: Optional[str] = None  # None for test samples


def load_novel(book_name: str) -> str:
    """
    Load novel text by book name.
    
    Args:
        book_name: Name of the book (e.g., "In Search of the Castaways")
    
    Returns:
        Full novel text as string
    """
    # Map book names to file names
    name_mapping = {
        "In Search of the Castaways": "In search of the castaways.txt",
        "The Count of Monte Cristo": "The Count of Monte Cristo.txt",
    }
    
    filename = name_mapping.get(book_name)
    if not filename:
        raise ValueError(f"Unknown book: {book_name}")
    
    filepath = DATASET_DIR / filename
    if not filepath.exists():
        raise FileNotFoundError(f"Novel file not found: {filepath}")
    
    return filepath.read_text(encoding='utf-8')


def load_training_data() -> list[Sample]:
    """Load all training samples from train.csv."""
    samples = []
    csv_path = DATASET_DIR / "train.csv"
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            samples.append(Sample(
                id=int(row['id']),
                book_name=row['book_name'],
                char=row['char'],
                caption=row['caption'],
                content=row['content'],
                label=row['label']
            ))
    
    return samples


def load_test_data() -> list[Sample]:
    """Load all test samples from test.csv."""
    samples = []
    csv_path = DATASET_DIR / "test.csv"
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            samples.append(Sample(
                id=int(row['id']),
                book_name=row['book_name'],
                char=row['char'],
                caption=row['caption'],
                content=row['content'],
                label=None  # Test samples don't have labels
            ))
    
    return samples


def get_balanced_samples(samples: list[Sample], n_consistent: int, n_contradict: int) -> list[Sample]:
    """
    Get a balanced subset of samples with specified counts.
    
    Args:
        samples: List of all samples
        n_consistent: Number of consistent samples to include
        n_contradict: Number of contradicting samples to include
    
    Returns:
        Balanced list of samples
    """
    consistent = [s for s in samples if s.label == 'consistent']
    contradict = [s for s in samples if s.label == 'contradict']
    
    if len(consistent) < n_consistent:
        raise ValueError(f"Not enough consistent samples: {len(consistent)} < {n_consistent}")
    if len(contradict) < n_contradict:
        raise ValueError(f"Not enough contradict samples: {len(contradict)} < {n_contradict}")
    
    return consistent[:n_consistent] + contradict[:n_contradict]


# Cache for novels (avoid re-reading)
_novel_cache: dict[str, str] = {}

def get_novel_text(book_name: str) -> str:
    """Get novel text with caching."""
    if book_name not in _novel_cache:
        _novel_cache[book_name] = load_novel(book_name)
    return _novel_cache[book_name]
