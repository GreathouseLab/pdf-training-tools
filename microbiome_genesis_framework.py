"""
Microbiome Genesis Framework
============================
Synthetic community simulator + deep learning model for inferring true
community composition from noisy sequencing data and aligning heterogeneous datasets.

Author: Greathouse Lab Collaboration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import random
from collections import defaultdict

# =============================================================================
# PART 1: CONFIGURATION AND DATA STRUCTURES
# =============================================================================

class SequencingProtocol(Enum):
    """Supported sequencing protocols with characteristic biases."""
    SHORT_READ_V4 = "16S_V4"           # ~250bp, Illumina
    SHORT_READ_V3V4 = "16S_V3V4"       # ~460bp, Illumina
    FULL_LENGTH_16S = "16S_FULL"       # ~1500bp, PacBio/ONT
    WGS_SHORT = "WGS_ILLUMINA"         # 150bp paired-end
    WGS_LONG = "WGS_NANOPORE"          # Variable length, higher error


@dataclass
class SimulatorConfig:
    """Configuration for the microbiome simulator."""
    # Reference database
    reference_db_path: str = "silva_138_16S.fasta"
    taxonomy_path: str = "silva_138_taxonomy.tsv"
    num_reference_sequences: int = 50000
    
    # Community structure
    min_species: int = 10
    max_species: int = 500
    abundance_distribution: str = "log_normal"  # or "dirichlet", "zero_inflated_ln"
    
    # Sequencing parameters
    min_library_size: int = 1000
    max_library_size: int = 100000
    
    # Artifact parameters
    pcr_cycles: int = 25
    pcr_efficiency_mean: float = 0.85
    pcr_efficiency_std: float = 0.1
    chimera_rate: float = 0.02
    dropout_rate: float = 0.05
    
    # Error profiles by platform
    illumina_error_rate: float = 0.001
    pacbio_error_rate: float = 0.01
    nanopore_error_rate: float = 0.05


@dataclass
class ModelConfig:
    """Configuration for the inference model."""
    # Input processing
    kmer_size: int = 6
    max_reads_per_sample: int = 10000
    max_read_length: int = 300
    
    # Architecture
    embedding_dim: int = 256
    num_transformer_layers: int = 6
    num_attention_heads: int = 8
    feedforward_dim: int = 1024
    dropout: float = 0.1
    
    # Output
    num_taxa: int = 5000  # Max taxonomic units to predict
    latent_dim: int = 128  # Community embedding dimension


# =============================================================================
# PART 2: REFERENCE DATABASE AND SEQUENCE UTILITIES
# =============================================================================

class ReferenceDatabase:
    """
    Manages reference 16S/WGS sequences and taxonomy.
    In production, load from SILVA, Greengenes, or custom database.
    """
    
    def __init__(self, config: SimulatorConfig):
        self.config = config
        self.sequences: Dict[str, str] = {}      # taxon_id -> sequence
        self.taxonomy: Dict[str, List[str]] = {} # taxon_id -> [k, p, c, o, f, g, s]
        self.taxon_ids: List[str] = []
        
        # Region coordinates for 16S (E. coli numbering)
        self.regions = {
            "V1": (69, 99),
            "V2": (137, 242),
            "V3": (433, 497),
            "V4": (576, 682),
            "V5": (822, 879),
            "V6": (986, 1043),
            "V7": (1117, 1173),
            "V8": (1243, 1294),
            "V9": (1435, 1465),
        }
        
    def load_database(self):
        """Load reference sequences and taxonomy from files."""
        # Pseudocode - in production, parse FASTA and taxonomy files
        print(f"Loading {self.config.num_reference_sequences} reference sequences...")
        
        # Simulate loading for demonstration
        for i in range(self.config.num_reference_sequences):
            taxon_id = f"OTU_{i:06d}"
            self.sequences[taxon_id] = self._generate_random_16s()
            self.taxonomy[taxon_id] = self._generate_random_taxonomy()
            self.taxon_ids.append(taxon_id)
    
    def _generate_random_16s(self, length: int = 1500) -> str:
        """Generate a random 16S-like sequence for testing."""
        # In production, load real sequences
        bases = ['A', 'C', 'G', 'T']
        # Add some conserved regions characteristic of 16S
        seq = ''.join(random.choices(bases, weights=[0.25, 0.25, 0.30, 0.20], k=length))
        return seq
    
    def _generate_random_taxonomy(self) -> List[str]:
        """Generate random taxonomy for testing."""
        # In production, load from taxonomy file
        return [
            f"k__Bacteria",
            f"p__Phylum{random.randint(1, 30)}",
            f"c__Class{random.randint(1, 100)}",
            f"o__Order{random.randint(1, 300)}",
            f"f__Family{random.randint(1, 1000)}",
            f"g__Genus{random.randint(1, 3000)}",
            f"s__Species{random.randint(1, 10000)}"
        ]
    
    def extract_region(self, sequence: str, region: str) -> str:
        """Extract a specific variable region from 16S sequence."""
        if region not in self.regions:
            return sequence
        start, end = self.regions[region]
        # Add some fuzziness to mimic primer binding variation
        start = max(0, start + random.randint(-5, 5))
        end = min(len(sequence), end + random.randint(-5, 5))
        return sequence[start:end]
    
    def get_region_for_protocol(self, sequence: str, protocol: SequencingProtocol) -> str:
        """Get appropriate sequence region for a given protocol."""
        if protocol == SequencingProtocol.SHORT_READ_V4:
            return self.extract_region(sequence, "V4")
        elif protocol == SequencingProtocol.SHORT_READ_V3V4:
            v3 = self.extract_region(sequence, "V3")
            v4 = self.extract_region(sequence, "V4")
            # Include intervening sequence
            return sequence[433:682]  # V3-V4 region
        elif protocol == SequencingProtocol.FULL_LENGTH_16S:
            return sequence
        else:  # WGS protocols return random genomic fragments
            return sequence  # Simplified; WGS would sample from whole genome


# =============================================================================
# PART 3: ARTIFACT SIMULATION
# =============================================================================

class ArtifactSimulator:
    """
    Simulates realistic sequencing artifacts including:
    - PCR amplification bias
    - Chimera formation
    - Sequencing errors
    - Coverage bias
    - Dropout
    - Library size variation
    """
    
    def __init__(self, config: SimulatorConfig):
        self.config = config
        
        # Precompute GC-content bias lookup
        self.gc_bias_curve = self._compute_gc_bias_curve()
        
        # Error profiles by position (simplified)
        self.illumina_error_profile = self._generate_illumina_error_profile()
        self.nanopore_error_profile = self._generate_nanopore_error_profile()
    
    def _compute_gc_bias_curve(self) -> np.ndarray:
        """
        PCR efficiency varies with GC content.
        Returns efficiency multiplier for GC% from 0-100.
        """
        gc_range = np.arange(101)
        # Optimal around 50% GC, drops off at extremes
        efficiency = np.exp(-0.001 * (gc_range - 50)**2)
        return efficiency
    
    def _generate_illumina_error_profile(self, max_length: int = 300) -> np.ndarray:
        """
        Illumina error rate increases toward read ends.
        Returns position-specific error rates.
        """
        positions = np.arange(max_length)
        base_rate = self.config.illumina_error_rate
        # Error rate increases quadratically toward end
        error_rates = base_rate * (1 + 0.01 * (positions / max_length)**2)
        return error_rates
    
    def _generate_nanopore_error_profile(self, max_length: int = 2000) -> np.ndarray:
        """Nanopore has more uniform but higher error rate."""
        return np.ones(max_length) * self.config.nanopore_error_rate
    
    def compute_gc_content(self, sequence: str) -> float:
        """Calculate GC content of a sequence."""
        if len(sequence) == 0:
            return 0.5
        gc = sum(1 for base in sequence.upper() if base in 'GC')
        return gc / len(sequence)
    
    def apply_pcr_bias(self, abundances: Dict[str, float], 
                       sequences: Dict[str, str]) -> Dict[str, float]:
        """
        Simulate PCR amplification bias based on:
        - GC content
        - Primer binding efficiency
        - Stochastic amplification noise
        """
        biased_abundances = {}
        
        for taxon_id, abundance in abundances.items():
            if taxon_id not in sequences:
                continue
                
            seq = sequences[taxon_id]
            gc = self.compute_gc_content(seq)
            gc_idx = int(gc * 100)
            
            # GC bias
            gc_efficiency = self.gc_bias_curve[gc_idx]
            
            # Stochastic PCR noise (multiplicative)
            pcr_noise = np.random.lognormal(0, 0.3)
            
            # Primer binding efficiency (sequence-specific)
            primer_efficiency = np.random.beta(
                self.config.pcr_efficiency_mean * 10,
                (1 - self.config.pcr_efficiency_mean) * 10
            )
            
            # Compound effect over PCR cycles
            amplification_factor = (
                gc_efficiency * primer_efficiency * pcr_noise
            ) ** (self.config.pcr_cycles / 25)  # Normalize to 25 cycles
            
            biased_abundances[taxon_id] = abundance * amplification_factor
        
        # Renormalize
        total = sum(biased_abundances.values())
        if total > 0:
            biased_abundances = {k: v/total for k, v in biased_abundances.items()}
        
        return biased_abundances
    
    def generate_chimera(self, seq1: str, seq2: str) -> str:
        """
        Generate a chimeric sequence from two parent sequences.
        Chimeras typically form at conserved regions during PCR.
        """
        # Find a reasonable breakpoint (simplified)
        min_len = min(len(seq1), len(seq2))
        breakpoint = random.randint(min_len // 4, 3 * min_len // 4)
        
        # Random which parent is 5' vs 3'
        if random.random() < 0.5:
            return seq1[:breakpoint] + seq2[breakpoint:min_len]
        else:
            return seq2[:breakpoint] + seq1[breakpoint:min_len]
    
    def apply_sequencing_errors(self, sequence: str, 
                                 protocol: SequencingProtocol) -> str:
        """Apply position-specific sequencing errors."""
        if protocol in [SequencingProtocol.WGS_LONG, SequencingProtocol.FULL_LENGTH_16S]:
            error_profile = self.nanopore_error_profile
            # Nanopore has more indels
            indel_ratio = 0.3
        else:
            error_profile = self.illumina_error_profile
            indel_ratio = 0.1
        
        bases = list(sequence)
        result = []
        
        for i, base in enumerate(bases):
            if i >= len(error_profile):
                error_rate = error_profile[-1]
            else:
                error_rate = error_profile[i]
            
            if random.random() < error_rate:
                error_type = random.random()
                if error_type < indel_ratio / 2:
                    # Deletion - skip this base
                    continue
                elif error_type < indel_ratio:
                    # Insertion - add random base before
                    result.append(random.choice('ACGT'))
                    result.append(base)
                else:
                    # Substitution
                    alternatives = [b for b in 'ACGT' if b != base]
                    result.append(random.choice(alternatives))
            else:
                result.append(base)
        
        return ''.join(result)
    
    def apply_dropout(self, abundances: Dict[str, float]) -> Dict[str, float]:
        """
        Simulate stochastic dropout of low-abundance taxa.
        More likely for rare taxa.
        """
        filtered = {}
        for taxon_id, abundance in abundances.items():
            # Dropout probability inversely related to abundance
            dropout_prob = self.config.dropout_rate * (1 - abundance) ** 2
            if random.random() > dropout_prob:
                filtered[taxon_id] = abundance
        
        # Renormalize
        total = sum(filtered.values())
        if total > 0:
            filtered = {k: v/total for k, v in filtered.items()}
        
        return filtered
    
    def sample_library_size(self) -> int:
        """Sample a realistic library size (number of reads)."""
        # Log-uniform distribution between min and max
        log_min = np.log(self.config.min_library_size)
        log_max = np.log(self.config.max_library_size)
        log_size = np.random.uniform(log_min, log_max)
        return int(np.exp(log_size))


# =============================================================================
# PART 4: COMMUNITY SIMULATOR
# =============================================================================

class CommunitySimulator:
    """
    Main simulator class that generates synthetic microbial communities
    with realistic abundance distributions and sequencing artifacts.
    """
    
    def __init__(self, config: SimulatorConfig):
        self.config = config
        self.ref_db = ReferenceDatabase(config)
        self.artifact_sim = ArtifactSimulator(config)
        
        # Load reference database
        self.ref_db.load_database()
    
    def sample_abundance_distribution(self, num_species: int) -> np.ndarray:
        """
        Sample relative abundances from specified distribution.
        Returns normalized abundance vector.
        """
        if self.config.abundance_distribution == "log_normal":
            # Log-normal is commonly observed in microbiome data
            abundances = np.random.lognormal(mean=0, sigma=2, size=num_species)
            
        elif self.config.abundance_distribution == "dirichlet":
            # Dirichlet with low concentration = few dominants
            alpha = np.ones(num_species) * 0.1
            abundances = np.random.dirichlet(alpha)
            
        elif self.config.abundance_distribution == "zero_inflated_ln":
            # Log-normal with additional zeros (sparse communities)
            abundances = np.random.lognormal(mean=0, sigma=2, size=num_species)
            # Zero out ~30% of taxa
            zero_mask = np.random.random(num_species) < 0.3
            abundances[zero_mask] = 0
            
        else:
            raise ValueError(f"Unknown distribution: {self.config.abundance_distribution}")
        
        # Normalize to relative abundances
        abundances = abundances / abundances.sum()
        return abundances
    
    def generate_community(self, 
                          protocol: SequencingProtocol,
                          num_species: Optional[int] = None
                          ) -> Dict:
        """
        Generate a complete synthetic community with:
        - True composition (ground truth labels)
        - Observed reads (with artifacts)
        - Metadata
        """
        # Sample number of species
        if num_species is None:
            num_species = random.randint(
                self.config.min_species, 
                self.config.max_species
            )
        
        # Select taxa for this community
        selected_taxa = random.sample(self.ref_db.taxon_ids, num_species)
        
        # Sample true abundances
        true_abundances = self.sample_abundance_distribution(num_species)
        true_composition = {
            taxon: abund 
            for taxon, abund in zip(selected_taxa, true_abundances)
            if abund > 0
        }
        
        # Get reference sequences
        sequences = {
            taxon: self.ref_db.sequences[taxon] 
            for taxon in true_composition.keys()
        }
        
        # Apply PCR bias
        pcr_biased = self.artifact_sim.apply_pcr_bias(true_composition, sequences)
        
        # Apply dropout
        post_dropout = self.artifact_sim.apply_dropout(pcr_biased)
        
        # Sample library size
        library_size = self.artifact_sim.sample_library_size()
        
        # Generate reads
        reads = []
        read_origins = []  # Track true origin for training
        
        # Calculate read counts per taxon
        read_counts = {
            taxon: int(abund * library_size)
            for taxon, abund in post_dropout.items()
        }
        
        for taxon, count in read_counts.items():
            seq = sequences.get(taxon, "")
            if not seq:
                continue
                
            # Get appropriate region for protocol
            region_seq = self.ref_db.get_region_for_protocol(seq, protocol)
            
            for _ in range(count):
                # Chance of chimera formation
                if random.random() < self.config.chimera_rate:
                    # Pick another taxon for chimera
                    other_taxa = [t for t in post_dropout.keys() if t != taxon]
                    if other_taxa:
                        other_taxon = random.choice(other_taxa)
                        other_seq = self.ref_db.get_region_for_protocol(
                            sequences[other_taxon], protocol
                        )
                        read_seq = self.artifact_sim.generate_chimera(region_seq, other_seq)
                        read_origins.append(f"chimera:{taxon}:{other_taxon}")
                    else:
                        read_seq = region_seq
                        read_origins.append(taxon)
                else:
                    read_seq = region_seq
                    read_origins.append(taxon)
                
                # Apply sequencing errors
                read_seq = self.artifact_sim.apply_sequencing_errors(read_seq, protocol)
                reads.append(read_seq)
        
        # Shuffle reads (they don't come in order)
        combined = list(zip(reads, read_origins))
        random.shuffle(combined)
        if combined:
            reads, read_origins = zip(*combined)
            reads = list(reads)
            read_origins = list(read_origins)
        
        return {
            "true_composition": true_composition,
            "observed_composition": post_dropout,
            "taxonomy": {t: self.ref_db.taxonomy[t] for t in true_composition.keys()},
            "reads": reads,
            "read_origins": read_origins,
            "protocol": protocol.value,
            "library_size": library_size,
            "num_species_true": len(true_composition),
            "num_species_observed": len(post_dropout),
            "chimera_rate_actual": sum(1 for o in read_origins if o.startswith("chimera")) / max(len(read_origins), 1)
        }


# =============================================================================
# PART 5: DATASET AND DATA LOADING
# =============================================================================

class SyntheticMicrobiomeDataset(Dataset):
    """
    PyTorch Dataset that generates synthetic communities on-the-fly
    or loads pre-generated communities.
    """
    
    def __init__(self, 
                 config: SimulatorConfig,
                 model_config: ModelConfig,
                 num_samples: int,
                 protocols: List[SequencingProtocol] = None,
                 pregenerate: bool = False):
        
        self.simulator = CommunitySimulator(config)
        self.model_config = model_config
        self.num_samples = num_samples
        self.protocols = protocols or list(SequencingProtocol)
        self.pregenerate = pregenerate
        
        # Build k-mer vocabulary
        self.kmer_to_idx = self._build_kmer_vocab()
        self.vocab_size = len(self.kmer_to_idx)
        
        # Build taxon vocabulary (for output)
        self.taxon_to_idx = {
            taxon: idx for idx, taxon 
            in enumerate(self.simulator.ref_db.taxon_ids)
        }
        
        # Pre-generate if requested (for reproducibility)
        self.pregenerated = []
        if pregenerate:
            print(f"Pre-generating {num_samples} communities...")
            for i in range(num_samples):
                protocol = random.choice(self.protocols)
                community = self.simulator.generate_community(protocol)
                self.pregenerated.append(community)
    
    def _build_kmer_vocab(self) -> Dict[str, int]:
        """Build vocabulary of all possible k-mers."""
        k = self.model_config.kmer_size
        bases = 'ACGT'
        vocab = {'<PAD>': 0, '<UNK>': 1}
        
        def generate_kmers(prefix, k):
            if k == 0:
                return [prefix]
            kmers = []
            for base in bases:
                kmers.extend(generate_kmers(prefix + base, k - 1))
            return kmers
        
        for kmer in generate_kmers('', k):
            vocab[kmer] = len(vocab)
        
        return vocab
    
    def sequence_to_kmers(self, sequence: str) -> List[int]:
        """Convert a sequence to k-mer indices."""
        k = self.model_config.kmer_size
        kmers = []
        for i in range(len(sequence) - k + 1):
            kmer = sequence[i:i+k].upper()
            idx = self.kmer_to_idx.get(kmer, self.kmer_to_idx['<UNK>'])
            kmers.append(idx)
        return kmers
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Get or generate community
        if self.pregenerate:
            community = self.pregenerated[idx]
        else:
            protocol = random.choice(self.protocols)
            community = self.simulator.generate_community(protocol)
        
        # Process reads into k-mer tensors
        reads = community['reads']
        max_reads = min(len(reads), self.model_config.max_reads_per_sample)
        
        # Sample reads if too many
        if len(reads) > max_reads:
            reads = random.sample(reads, max_reads)
        
        # Convert reads to k-mer indices
        max_kmers = self.model_config.max_read_length - self.model_config.kmer_size + 1
        read_tensor = torch.zeros(max_reads, max_kmers, dtype=torch.long)
        read_mask = torch.zeros(max_reads, max_kmers, dtype=torch.bool)
        
        for i, read in enumerate(reads):
            kmers = self.sequence_to_kmers(read)[:max_kmers]
            read_tensor[i, :len(kmers)] = torch.tensor(kmers)
            read_mask[i, :len(kmers)] = True
        
        # Create ground truth composition vector
        true_composition = torch.zeros(self.model_config.num_taxa)
        for taxon, abundance in community['true_composition'].items():
            if taxon in self.taxon_to_idx:
                idx = self.taxon_to_idx[taxon]
                if idx < self.model_config.num_taxa:
                    true_composition[idx] = abundance
        
        # Normalize (in case some taxa were out of range)
        if true_composition.sum() > 0:
            true_composition = true_composition / true_composition.sum()
        
        # Protocol one-hot encoding
        protocol_idx = list(SequencingProtocol).index(
            SequencingProtocol(community['protocol'])
        )
        protocol_onehot = F.one_hot(
            torch.tensor(protocol_idx), 
            num_classes=len(SequencingProtocol)
        ).float()
        
        return {
            'reads': read_tensor,           # [max_reads, max_kmers]
            'read_mask': read_mask,         # [max_reads, max_kmers]
            'true_composition': true_composition,  # [num_taxa]
            'protocol': protocol_onehot,    # [num_protocols]
            'num_reads': torch.tensor(len(reads)),
            'library_size': torch.tensor(community['library_size']),
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Custom collate function for batching communities."""
    return {
        'reads': torch.stack([b['reads'] for b in batch]),
        'read_mask': torch.stack([b['read_mask'] for b in batch]),
        'true_composition': torch.stack([b['true_composition'] for b in batch]),
        'protocol': torch.stack([b['protocol'] for b in batch]),
        'num_reads': torch.stack([b['num_reads'] for b in batch]),
        'library_size': torch.stack([b['library_size'] for b in batch]),
    }


# =============================================================================
# PART 6: MODEL ARCHITECTURES
# =============================================================================

class KmerEmbedding(nn.Module):
    """Embedding layer for k-mers with optional positional encoding."""
    
    def __init__(self, vocab_size: int, embedding_dim: int, max_length: int = 1000):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.pos_encoding = nn.Parameter(torch.randn(1, max_length, embedding_dim) * 0.02)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, num_reads, seq_len]
        embedded = self.embedding(x)  # [batch, num_reads, seq_len, embed_dim]
        seq_len = x.shape[-1]
        embedded = embedded + self.pos_encoding[:, :seq_len, :]
        return embedded


class ReadEncoder(nn.Module):
    """
    Encodes individual reads using 1D CNN or Transformer.
    Outputs a fixed-size vector per read.
    """
    
    def __init__(self, config: ModelConfig, vocab_size: int, use_transformer: bool = True):
        super().__init__()
        self.config = config
        self.use_transformer = use_transformer
        
        self.kmer_embedding = KmerEmbedding(vocab_size, config.embedding_dim)
        
        if use_transformer:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=config.embedding_dim,
                nhead=config.num_attention_heads,
                dim_feedforward=config.feedforward_dim,
                dropout=config.dropout,
                batch_first=True
            )
            self.encoder = nn.TransformerEncoder(
                encoder_layer, 
                num_layers=config.num_transformer_layers // 2
            )
        else:
            # 1D CNN alternative
            self.conv_layers = nn.Sequential(
                nn.Conv1d(config.embedding_dim, config.embedding_dim, 3, padding=1),
                nn.ReLU(),
                nn.Conv1d(config.embedding_dim, config.embedding_dim, 3, padding=1),
                nn.ReLU(),
                nn.Conv1d(config.embedding_dim, config.embedding_dim, 3, padding=1),
                nn.ReLU(),
            )
        
        self.output_proj = nn.Linear(config.embedding_dim, config.embedding_dim)
    
    def forward(self, reads: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            reads: [batch, num_reads, seq_len] k-mer indices
            mask: [batch, num_reads, seq_len] boolean mask
        
        Returns:
            read_embeddings: [batch, num_reads, embedding_dim]
        """
        batch_size, num_reads, seq_len = reads.shape
        
        # Flatten for processing
        reads_flat = reads.view(batch_size * num_reads, seq_len)
        mask_flat = mask.view(batch_size * num_reads, seq_len)
        
        # Embed k-mers
        embedded = self.kmer_embedding(reads_flat)  # [B*R, S, E]
        
        if self.use_transformer:
            # Transformer encoding with attention mask
            attn_mask = ~mask_flat  # Transformer uses inverted mask
            encoded = self.encoder(embedded, src_key_padding_mask=attn_mask)
        else:
            # CNN encoding
            embedded = embedded.transpose(1, 2)  # [B*R, E, S]
            encoded = self.conv_layers(embedded)
            encoded = encoded.transpose(1, 2)  # [B*R, S, E]
        
        # Pool over sequence (masked mean)
        mask_expanded = mask_flat.unsqueeze(-1).float()
        pooled = (encoded * mask_expanded).sum(dim=1) / (mask_expanded.sum(dim=1) + 1e-8)
        
        # Project
        output = self.output_proj(pooled)
        
        # Reshape back
        output = output.view(batch_size, num_reads, -1)
        return output


class CommunityEncoder(nn.Module):
    """
    Aggregates read embeddings into a community-level representation.
    Uses attention-based pooling to weight reads.
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Cross-attention between reads
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.embedding_dim,
            nhead=config.num_attention_heads,
            dim_feedforward=config.feedforward_dim,
            dropout=config.dropout,
            batch_first=True
        )
        self.cross_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_transformer_layers // 2
        )
        
        # Attention-based pooling
        self.attention_weights = nn.Sequential(
            nn.Linear(config.embedding_dim, config.embedding_dim // 4),
            nn.Tanh(),
            nn.Linear(config.embedding_dim // 4, 1)
        )
        
        # Protocol conditioning
        self.protocol_proj = nn.Linear(len(SequencingProtocol), config.embedding_dim)
        
        # Output projections
        self.latent_proj = nn.Linear(config.embedding_dim, config.latent_dim)
    
    def forward(self, read_embeddings: torch.Tensor, 
                protocol: torch.Tensor,
                num_reads: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            read_embeddings: [batch, num_reads, embedding_dim]
            protocol: [batch, num_protocols] one-hot
            num_reads: [batch] actual number of reads
        
        Returns:
            community_embedding: [batch, embedding_dim]
            latent_embedding: [batch, latent_dim] for alignment
        """
        batch_size, max_reads, embed_dim = read_embeddings.shape
        
        # Create mask for valid reads
        read_indices = torch.arange(max_reads, device=read_embeddings.device)
        read_mask = read_indices.unsqueeze(0) < num_reads.unsqueeze(1)  # [B, R]
        
        # Add protocol conditioning
        protocol_embed = self.protocol_proj(protocol)  # [B, E]
        # Add as a special token (prepend)
        protocol_token = protocol_embed.unsqueeze(1)  # [B, 1, E]
        augmented = torch.cat([protocol_token, read_embeddings], dim=1)  # [B, R+1, E]
        
        # Extend mask
        protocol_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=read_mask.device)
        augmented_mask = torch.cat([protocol_mask, read_mask], dim=1)
        
        # Cross-attention between reads
        attn_mask = ~augmented_mask
        encoded = self.cross_encoder(augmented, src_key_padding_mask=attn_mask)
        
        # Attention-based pooling (exclude protocol token)
        read_encoded = encoded[:, 1:, :]  # [B, R, E]
        attention_scores = self.attention_weights(read_encoded).squeeze(-1)  # [B, R]
        attention_scores = attention_scores.masked_fill(~read_mask, float('-inf'))
        attention_weights = F.softmax(attention_scores, dim=-1)  # [B, R]
        
        # Weighted sum
        community_embedding = torch.bmm(
            attention_weights.unsqueeze(1), 
            read_encoded
        ).squeeze(1)  # [B, E]
        
        # Latent projection for alignment
        latent_embedding = self.latent_proj(community_embedding)  # [B, L]
        latent_embedding = F.normalize(latent_embedding, dim=-1)  # Unit sphere
        
        return community_embedding, latent_embedding


class CompositionDecoder(nn.Module):
    """
    Decodes community embedding into taxonomic composition.
    Outputs probability distribution over taxa.
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        self.decoder = nn.Sequential(
            nn.Linear(config.embedding_dim, config.feedforward_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.feedforward_dim, config.feedforward_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.feedforward_dim, config.num_taxa),
        )
    
    def forward(self, community_embedding: torch.Tensor) -> torch.Tensor:
        """
        Args:
            community_embedding: [batch, embedding_dim]
        
        Returns:
            composition: [batch, num_taxa] (softmax probabilities)
        """
        logits = self.decoder(community_embedding)
        composition = F.softmax(logits, dim=-1)
        return composition


class MicrobiomeGenesisModel(nn.Module):
    """
    Full model: Raw reads -> True composition + Latent embedding
    
    Architecture:
    1. ReadEncoder: Individual reads -> read embeddings
    2. CommunityEncoder: Read embeddings -> community embedding + latent
    3. CompositionDecoder: Community embedding -> composition
    """
    
    def __init__(self, config: ModelConfig, vocab_size: int, use_transformer: bool = True):
        super().__init__()
        self.config = config
        
        self.read_encoder = ReadEncoder(config, vocab_size, use_transformer)
        self.community_encoder = CommunityEncoder(config)
        self.composition_decoder = CompositionDecoder(config)
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Full forward pass.
        
        Args:
            batch: Dictionary with 'reads', 'read_mask', 'protocol', 'num_reads'
        
        Returns:
            Dictionary with 'predicted_composition', 'latent_embedding'
        """
        # Encode reads
        read_embeddings = self.read_encoder(batch['reads'], batch['read_mask'])
        
        # Encode community
        community_embedding, latent_embedding = self.community_encoder(
            read_embeddings, 
            batch['protocol'],
            batch['num_reads']
        )
        
        # Decode composition
        predicted_composition = self.composition_decoder(community_embedding)
        
        return {
            'predicted_composition': predicted_composition,
            'community_embedding': community_embedding,
            'latent_embedding': latent_embedding,
        }
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# PART 7: LOSS FUNCTIONS
# =============================================================================

class CompositionLoss(nn.Module):
    """
    Combined loss for composition prediction:
    - KL divergence (primary)
    - Bray-Curtis dissimilarity (ecological)
    - Sparsity regularization
    """
    
    def __init__(self, kl_weight: float = 1.0, 
                 bray_curtis_weight: float = 0.5,
                 sparsity_weight: float = 0.1):
        super().__init__()
        self.kl_weight = kl_weight
        self.bray_curtis_weight = bray_curtis_weight
        self.sparsity_weight = sparsity_weight
    
    def forward(self, predicted: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            predicted: [batch, num_taxa] predicted composition
            target: [batch, num_taxa] true composition
        """
        # KL divergence (target as reference)
        # Add small epsilon for numerical stability
        eps = 1e-8
        kl_loss = F.kl_div(
            (predicted + eps).log(),
            target + eps,
            reduction='batchmean'
        )
        
        # Bray-Curtis dissimilarity
        # BC = sum(|p - t|) / sum(p + t)
        numerator = torch.abs(predicted - target).sum(dim=-1)
        denominator = (predicted + target).sum(dim=-1) + eps
        bray_curtis = (numerator / denominator).mean()
        
        # Sparsity: encourage similar sparsity to target
        pred_entropy = -(predicted * (predicted + eps).log()).sum(dim=-1)
        target_entropy = -(target * (target + eps).log()).sum(dim=-1)
        sparsity_loss = F.mse_loss(pred_entropy, target_entropy)
        
        total = (
            self.kl_weight * kl_loss + 
            self.bray_curtis_weight * bray_curtis +
            self.sparsity_weight * sparsity_loss
        )
        
        return {
            'total': total,
            'kl': kl_loss,
            'bray_curtis': bray_curtis,
            'sparsity': sparsity_loss
        }


class AlignmentLoss(nn.Module):
    """
    Contrastive loss for aligning communities across protocols.
    Same community sequenced with different protocols should have
    similar latent embeddings.
    """
    
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, embeddings: torch.Tensor, 
                labels: torch.Tensor) -> torch.Tensor:
        """
        InfoNCE-style contrastive loss.
        
        Args:
            embeddings: [batch, latent_dim] normalized embeddings
            labels: [batch] community IDs (same ID = positive pair)
        """
        # Compute similarity matrix
        similarity = torch.mm(embeddings, embeddings.t()) / self.temperature
        
        # Create positive mask (same community = positive)
        labels = labels.unsqueeze(1)
        positive_mask = (labels == labels.t()).float()
        
        # Remove self-similarity
        identity = torch.eye(len(embeddings), device=embeddings.device)
        positive_mask = positive_mask - identity
        
        # InfoNCE loss
        exp_sim = torch.exp(similarity)
        
        # Denominator: sum over all negatives
        neg_mask = 1 - positive_mask - identity
        neg_sum = (exp_sim * neg_mask).sum(dim=1, keepdim=True)
        
        # Log probability of positives
        log_prob = similarity - torch.log(exp_sim + neg_sum + 1e-8)
        
        # Average over positives
        num_positives = positive_mask.sum(dim=1)
        loss = -(log_prob * positive_mask).sum(dim=1) / (num_positives + 1e-8)
        
        return loss.mean()


# =============================================================================
# PART 8: TRAINING LOOP
# =============================================================================

class Trainer:
    """Training loop for the Microbiome Genesis Model."""
    
    def __init__(self,
                 model: MicrobiomeGenesisModel,
                 train_dataset: SyntheticMicrobiomeDataset,
                 val_dataset: SyntheticMicrobiomeDataset,
                 config: Dict):
        
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        
        # Losses
        self.composition_loss = CompositionLoss()
        self.alignment_loss = AlignmentLoss()
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.get('learning_rate', 1e-4),
            weight_decay=config.get('weight_decay', 0.01)
        )
        
        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.get('num_epochs', 100)
        )
        
        # DataLoaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.get('batch_size', 16),
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=config.get('num_workers', 4),
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=config.get('batch_size', 16),
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=config.get('num_workers', 4),
            pin_memory=True
        )
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        total_kl = 0
        total_bc = 0
        num_batches = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(batch)
            
            # Compute losses
            comp_losses = self.composition_loss(
                outputs['predicted_composition'],
                batch['true_composition']
            )
            
            loss = comp_losses['total']
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Logging
            total_loss += loss.item()
            total_kl += comp_losses['kl'].item()
            total_bc += comp_losses['bray_curtis'].item()
            num_batches += 1
            
            if batch_idx % 100 == 0:
                print(f"  Batch {batch_idx}/{len(self.train_loader)}, "
                      f"Loss: {loss.item():.4f}, BC: {comp_losses['bray_curtis'].item():.4f}")
        
        return {
            'train_loss': total_loss / num_batches,
            'train_kl': total_kl / num_batches,
            'train_bray_curtis': total_bc / num_batches
        }
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validation pass."""
        self.model.eval()
        total_loss = 0
        total_bc = 0
        num_batches = 0
        
        for batch in self.val_loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            outputs = self.model(batch)
            
            comp_losses = self.composition_loss(
                outputs['predicted_composition'],
                batch['true_composition']
            )
            
            total_loss += comp_losses['total'].item()
            total_bc += comp_losses['bray_curtis'].item()
            num_batches += 1
        
        return {
            'val_loss': total_loss / num_batches,
            'val_bray_curtis': total_bc / num_batches
        }
    
    def train(self, num_epochs: int):
        """Full training loop."""
        best_val_loss = float('inf')
        
        print(f"Training on {self.device}")
        print(f"Model parameters: {self.model.count_parameters():,}")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Train
            train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.validate()
            
            # Learning rate scheduling
            self.scheduler.step()
            
            # Logging
            print(f"  Train Loss: {train_metrics['train_loss']:.4f}, "
                  f"BC: {train_metrics['train_bray_curtis']:.4f}")
            print(f"  Val Loss: {val_metrics['val_loss']:.4f}, "
                  f"BC: {val_metrics['val_bray_curtis']:.4f}")
            
            # Save best model
            if val_metrics['val_loss'] < best_val_loss:
                best_val_loss = val_metrics['val_loss']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_metrics['val_loss'],
                }, 'best_model.pt')
                print("  Saved best model!")


# =============================================================================
# PART 9: SCALING ESTIMATES AND RECOMMENDATIONS
# =============================================================================

SCALING_ESTIMATES = """
================================================================================
SCALING ESTIMATES FOR MICROBIOME GENESIS MODEL
================================================================================

┌─────────────────────┬─────────────────┬─────────────────┬─────────────────────┐
│ Parameter           │ Toy Prototype   │ Intermediate    │ Genesis-Ready       │
├─────────────────────┼─────────────────┼─────────────────┼─────────────────────┤
│ TRAINING DATA                                                                 │
├─────────────────────┼─────────────────┼─────────────────┼─────────────────────┤
│ Synthetic communities│ 10,000         │ 500,000         │ 10,000,000          │
│ Reads per community │ 1,000-10,000    │ 5,000-50,000    │ 10,000-100,000      │
│ Reference DB size   │ 5,000 seqs      │ 50,000 seqs     │ 500,000 seqs        │
│ Protocols simulated │ 2               │ 4               │ 5+                  │
│ Dataset size (disk) │ ~5 GB           │ ~500 GB         │ ~5 TB               │
├─────────────────────┼─────────────────┼─────────────────┼─────────────────────┤
│ MODEL ARCHITECTURE                                                            │
├─────────────────────┼─────────────────┼─────────────────┼─────────────────────┤
│ K-mer size          │ 6               │ 6               │ 8                   │
│ Vocab size          │ 4,098           │ 4,098           │ 65,538              │
│ Embedding dim       │ 128             │ 256             │ 512                 │
│ Transformer layers  │ 4               │ 8               │ 16                  │
│ Attention heads     │ 4               │ 8               │ 16                  │
│ Feedforward dim     │ 512             │ 1,024           │ 2,048               │
│ Max reads/sample    │ 1,000           │ 5,000           │ 20,000              │
│ Output taxa         │ 1,000           │ 5,000           │ 50,000              │
│ Latent dim          │ 64              │ 128             │ 256                 │
│ Total parameters    │ ~5M             │ ~50M            │ ~500M               │
├─────────────────────┼─────────────────┼─────────────────┼─────────────────────┤
│ TRAINING COMPUTE                                                              │
├─────────────────────┼─────────────────┼─────────────────┼─────────────────────┤
│ Batch size          │ 32              │ 64              │ 256 (gradient accum)│
│ Training epochs     │ 20              │ 50              │ 100                 │
│ GPU type            │ RTX 3090 (24GB) │ A100 (40GB)     │ 8x A100 (80GB)      │
│ GPU hours           │ ~10 hrs         │ ~500 hrs        │ ~10,000 hrs         │
│ Wall clock time     │ ~10 hrs         │ ~2 weeks        │ ~2 months           │
│ Estimated cost      │ ~$30            │ ~$2,000         │ ~$30,000-50,000     │
├─────────────────────┼─────────────────┼─────────────────┼─────────────────────┤
│ EXPECTED PERFORMANCE                                                          │
├─────────────────────┼─────────────────┼─────────────────┼─────────────────────┤
│ Bray-Curtis error   │ ~0.3            │ ~0.15           │ ~0.05               │
│ Genus-level accuracy│ ~70%            │ ~85%            │ ~95%                │
│ Cross-protocol align│ Moderate        │ Good            │ Excellent           │
│ Chimera detection   │ Basic           │ Good            │ State-of-art        │
└─────────────────────┴─────────────────┴─────────────────┴─────────────────────┘

RECOMMENDED PROGRESSION:
------------------------
1. TOY PROTOTYPE (1-2 weeks)
   - Validate simulator produces realistic artifacts
   - Confirm model architecture converges
   - Establish baseline metrics
   - Test on small subset of real data
   
2. INTERMEDIATE (1-2 months)
   - Scale to meaningful reference database (SILVA subset)
   - Extensive hyperparameter tuning
   - Add contrastive learning for protocol alignment
   - Validate on public datasets (HMP, Earth Microbiome Project)
   - Publish preprint with promising results
   
3. GENESIS-READY (6-12 months)
   - Full SILVA/GTDB reference coverage
   - Multi-task learning (composition + diversity + taxa detection)
   - Domain adaptation for real data
   - Fine-tuning protocol for lab-specific models
   - Integration with existing pipelines (QIIME2, DADA2)

KEY TECHNICAL RECOMMENDATIONS:
------------------------------
• Use on-the-fly simulation during training (saves disk, adds variation)
• Implement gradient checkpointing for large models
• Mixed precision training (FP16/BF16) for 2x speedup
• Consider sequence-to-sequence formulation for rare taxa
• Add auxiliary tasks: chimera classification, protocol prediction
• Use curriculum learning: simple→complex communities
• Validate extensively on held-out real datasets before scaling
"""


# =============================================================================
# PART 10: MAIN EXECUTION
# =============================================================================

def create_toy_model():
    """Create a toy prototype for testing."""
    sim_config = SimulatorConfig(
        num_reference_sequences=5000,
        min_species=10,
        max_species=100,
        min_library_size=1000,
        max_library_size=10000,
    )
    
    model_config = ModelConfig(
        kmer_size=6,
        max_reads_per_sample=1000,
        embedding_dim=128,
        num_transformer_layers=4,
        num_attention_heads=4,
        feedforward_dim=512,
        num_taxa=1000,
        latent_dim=64,
    )
    
    return sim_config, model_config


def create_intermediate_model():
    """Create intermediate research model."""
    sim_config = SimulatorConfig(
        num_reference_sequences=50000,
        min_species=20,
        max_species=300,
        min_library_size=5000,
        max_library_size=50000,
    )
    
    model_config = ModelConfig(
        kmer_size=6,
        max_reads_per_sample=5000,
        embedding_dim=256,
        num_transformer_layers=8,
        num_attention_heads=8,
        feedforward_dim=1024,
        num_taxa=5000,
        latent_dim=128,
    )
    
    return sim_config, model_config


def main():
    """Main execution for toy prototype."""
    print("=" * 80)
    print("MICROBIOME GENESIS FRAMEWORK - TOY PROTOTYPE")
    print("=" * 80)
    
    # Create configurations
    sim_config, model_config = create_toy_model()
    
    # Create datasets
    print("\nCreating training dataset...")
    train_dataset = SyntheticMicrobiomeDataset(
        sim_config, 
        model_config, 
        num_samples=1000,
        protocols=[SequencingProtocol.SHORT_READ_V4, SequencingProtocol.SHORT_READ_V3V4],
        pregenerate=False  # Generate on-the-fly
    )
    
    print("Creating validation dataset...")
    val_dataset = SyntheticMicrobiomeDataset(
        sim_config,
        model_config,
        num_samples=200,
        protocols=[SequencingProtocol.SHORT_READ_V4, SequencingProtocol.SHORT_READ_V3V4],
        pregenerate=True  # Fixed for reproducibility
    )
    
    # Create model
    print("\nInitializing model...")
    model = MicrobiomeGenesisModel(
        model_config, 
        vocab_size=train_dataset.vocab_size,
        use_transformer=True
    )
    print(f"Model parameters: {model.count_parameters():,}")
    
    # Training configuration
    train_config = {
        'learning_rate': 1e-4,
        'weight_decay': 0.01,
        'batch_size': 16,
        'num_epochs': 20,
        'num_workers': 0,  # Set to 4+ in production
    }
    
    # Create trainer
    trainer = Trainer(model, train_dataset, val_dataset, train_config)
    
    # Train!
    print("\nStarting training...")
    trainer.train(num_epochs=train_config['num_epochs'])
    
    print("\n" + SCALING_ESTIMATES)


if __name__ == "__main__":
    main()
