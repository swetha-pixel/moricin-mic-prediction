"""
Generate T5XL Embeddings for Truncated Antimicrobial Peptide Sequences (40 AA)
"""

import torch
import numpy as np
from transformers import T5Tokenizer, T5EncoderModel
from Bio import SeqIO
import os

def generate_t5xl_embeddings(fasta_file, output_npy, max_length=40):
    """
    Generate T5XL embeddings for Antimicrobial Peptide peptides
    
    Parameters:
    -----------
    fasta_file : str
        Path to FASTA file with truncated sequences (40 AA each)
    output_npy : str
        Path to save embeddings as .npy file
    max_length : int
        Maximum sequence length (default: 40)
    
    Returns:
    --------
    numpy array with shape (n_sequences, 40, 1024)
    """
    
    print("="*70)
    print("T5XL EMBEDDING GENERATION FOR Antimicrobial Peptide")
    print("="*70)
    
    # Check if file exists
    if not os.path.exists(fasta_file):
        print(f"\n❌ ERROR: File not found: {fasta_file}")
        print("\nPlease check:")
        print("  1. Is the file path correct?")
        print("  2. Did you run the truncation script first?")
        return None
    
    print(f"\n📁 Input file: {fasta_file}")
    print(f"📁 Output file: {output_npy}")
    
    # Load T5-XL model
    print(f"\n Loading ProtT5-XL model...")
    print("   (This may take a few minutes on first run)")
    
    try:
        tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", do_lower_case=False)
        model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50")
        
        # Use GPU if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model = model.eval()
        
        print(f"   ✓ Model loaded successfully")
        print(f"   ✓ Using device: {device}")
        
    except Exception as e:
        print(f"\n❌ ERROR loading model: {e}")
        print("\nPlease install required packages:")
        print("  pip install transformers torch biopython")
        return None
    
    # Count sequences
    num_sequences = sum(1 for _ in SeqIO.parse(fasta_file, "fasta"))
    print(f"\n Found {num_sequences} sequences to process")
    
    # Process sequences
    print(f"\n Generating embeddings...")
    print("-"*70)
    
    embeddings = []
    sequence_ids = []
    
    for i, record in enumerate(SeqIO.parse(fasta_file, "fasta")):
        # Get sequence
        sequence = str(record.seq).upper().replace(" ", "")
        
        # Ensure exactly 40 AA
        if len(sequence) > max_length:
            print(f"   ⚠️  Warning: Sequence {i+1} is {len(sequence)} AA, truncating to {max_length}")
            sequence = sequence[:max_length]
        elif len(sequence) < max_length:
            print(f"   ⚠️  Warning: Sequence {i+1} is {len(sequence)} AA, padding to {max_length}")
            sequence = sequence.ljust(max_length, 'X')
        
        # Add spaces between amino acids (required by T5)
        spaced_sequence = ' '.join(list(sequence))
        
        # Tokenize
        with torch.no_grad():
            ids = tokenizer(
                spaced_sequence,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=max_length + 2  # +2 for special tokens <CLS> and <EOS>
            )
            
            # Move to device
            ids = {k: v.to(device) for k, v in ids.items()}
            
            # Generate embedding
            output = model(**ids)
            
            # Get embedding: shape [1, seq_len+2, 1024]
            embedding = output.last_hidden_state
            
            # Remove batch dimension and move to CPU
            embedding = embedding.squeeze(0).cpu().numpy()
            
            # Remove special tokens (keep only positions 1 to 40)
            # Position 0 is <CLS> token, position 41 is <EOS> token
            embedding = embedding[1:max_length+1, :]  # Shape: [40, 1024]
            
            # CRITICAL: Do NOT add 41st row!
            # The genomic features will be added later by Genomesequence_concat()
            
            embeddings.append(embedding)
            sequence_ids.append(record.id)
        
        # Progress indicator
        if (i + 1) % 10 == 0:
            print(f"   Processed {i+1}/{num_sequences} sequences...")
    
    # Convert to numpy array
    embeddings = np.array(embeddings, dtype=np.float32)
    
    print(f"\n✅ Embedding generation complete!")
    print("-"*70)
    print(f"   Final shape: {embeddings.shape}")
    print(f"   Expected: ({num_sequences}, 40, 1024)")
    
    # Verify shape
    assert embeddings.shape[1] == max_length, f"❌ Wrong sequence length: {embeddings.shape[1]}, expected {max_length}"
    assert embeddings.shape[2] == 1024, f"❌ Wrong embedding dimension: {embeddings.shape[2]}, expected 1024"
    
    print(f"   ✓ Shape verification passed!")
    
    # Check for issues
    if np.isnan(embeddings).any():
        nan_count = np.isnan(embeddings).sum()
        print(f"   ⚠️  WARNING: Found {nan_count} NaN values")
    else:
        print(f"   ✓ No NaN values")
    
    if np.isinf(embeddings).any():
        inf_count = np.isinf(embeddings).sum()
        print(f"   ⚠️  WARNING: Found {inf_count} Inf values")
    else:
        print(f"   ✓ No Inf values")
    
    # Statistics
    print(f"\n Embedding statistics:")
    print(f"   Min value: {embeddings.min():.6f}")
    print(f"   Max value: {embeddings.max():.6f}")
    print(f"   Mean: {embeddings.mean():.6f}")
    print(f"   Std: {embeddings.std():.6f}")
    
    # Save embeddings
    print(f"\n Saving embeddings...")
    np.save(output_npy, embeddings)
    
    # Verify saved file
    loaded = np.load(output_npy)
    if np.array_equal(loaded, embeddings):
        print(f"   ✓ Saved successfully: {output_npy}")
        print(f"   ✓ File size: {os.path.getsize(output_npy) / (1024*1024):.2f} MB")
    else:
        print(f"   ❌ ERROR: Verification failed!")
        return None
    
    print("\n" + "="*70)
    print("✅ T5XL EMBEDDINGS READY!")
    print("="*70)
    
    print(f"\n Output file: {output_npy}")
    print(f"   Shape: {embeddings.shape}")
    print(f"   Data type: {embeddings.dtype}")
    
    print(f"\n Next step:")
    print(f"   Use this file in your MIC prediction script")
    print(f"   Load with: np.load('{output_npy}')")
    
    return embeddings


def verify_embeddings(npy_file, expected_sequences=None):
    """
    Verify that embeddings are correct
    """
    
    print("\n" + "="*70)
    print("EMBEDDING VERIFICATION")
    print("="*70)
    
    if not os.path.exists(npy_file):
        print(f"❌ File not found: {npy_file}")
        return False
    
    # Load embeddings
    embeddings = np.load(npy_file)
    
    print(f"\n✓ Loaded: {npy_file}")
    print(f"  Shape: {embeddings.shape}")
    print(f"  Data type: {embeddings.dtype}")
    print(f"  File size: {os.path.getsize(npy_file) / (1024*1024):.2f} MB")
    
    # Check shape
    n_sequences, seq_len, embed_dim = embeddings.shape
    
    checks_passed = 0
    total_checks = 0
    
    # Check 1: Sequence length
    total_checks += 1
    if seq_len == 40:
        print(f"\n✅ Check 1: Sequence length = {seq_len} (CORRECT)")
        checks_passed += 1
    else:
        print(f"\n❌ Check 1: Sequence length = {seq_len} (Expected 40)")
    
    # Check 2: Embedding dimension
    total_checks += 1
    if embed_dim == 1024:
        print(f"✅ Check 2: Embedding dimension = {embed_dim} (CORRECT)")
        checks_passed += 1
    else:
        print(f"❌ Check 2: Embedding dimension = {embed_dim} (Expected 1024)")
    
    # Check 3: Number of sequences
    total_checks += 1
    if expected_sequences is not None:
        if n_sequences == expected_sequences:
            print(f"✅ Check 3: Number of sequences = {n_sequences} (CORRECT)")
            checks_passed += 1
        else:
            print(f"❌ Check 3: Number of sequences = {n_sequences} (Expected {expected_sequences})")
    else:
        print(f"ℹ️  Check 3: Number of sequences = {n_sequences}")
        checks_passed += 1
    
    # Check 4: No NaN values
    total_checks += 1
    if not np.isnan(embeddings).any():
        print(f"✅ Check 4: No NaN values")
        checks_passed += 1
    else:
        print(f"❌ Check 4: Contains NaN values")
    
    # Check 5: No Inf values
    total_checks += 1
    if not np.isinf(embeddings).any():
        print(f"✅ Check 5: No Inf values")
        checks_passed += 1
    else:
        print(f"❌ Check 5: Contains Inf values")
    
    # Check 6: Reasonable value range
    total_checks += 1
    if -50 < embeddings.min() < 50 and -50 < embeddings.max() < 50:
        print(f"✅ Check 6: Value range is reasonable ({embeddings.min():.2f} to {embeddings.max():.2f})")
        checks_passed += 1
    else:
        print(f"⚠️  Check 6: Value range unusual ({embeddings.min():.2f} to {embeddings.max():.2f})")
        checks_passed += 1  # Not critical
    
    # Summary
    print(f"\n" + "="*70)
    print(f"VERIFICATION RESULT: {checks_passed}/{total_checks} checks passed")
    print("="*70)
    
    if checks_passed == total_checks:
        print("\n✅ ALL CHECKS PASSED! Embeddings are ready for MIC prediction.")
        return True
    else:
        print(f"\n⚠️  {total_checks - checks_passed} check(s) failed. Please review above.")
        return False


# ============================================================
# MAIN EXECUTION - UPDATE YOUR FILE PATHS HERE
# ============================================================

if __name__ == "__main__":
    
    print("="*70)
    print("T5XL EMBEDDING GENERATION PIPELINE")
    print("="*70)
    
    # ╔════════════════════════════════════════════════════════════╗
    # ║  UPDATE THESE FILE PATHS WITH YOUR ACTUAL FILES           ║
    # ╚════════════════════════════════════════════════════════════╝
    
    # INPUT: Your truncated FASTA file (from truncation script)
    FASTA_FILE = 'Antimicrobial Peptide_truncated_40.fasta'  # ← CHANGE THIS
    
    # OUTPUT: Where to save T5XL embeddings
    OUTPUT_NPY = 'Antimicrobial Peptide_T5XL_40.npy'  # ← CHANGE THIS (optional)
    
    # How many sequences do you expect? (optional, for verification)
    EXPECTED_SEQUENCES = None  # ← Set to your number, e.g., 52
    
    # ╔════════════════════════════════════════════════════════════╗
    # ║  RUN EMBEDDING GENERATION                                  ║
    # ╚════════════════════════════════════════════════════════════╝
    
    try:
        # Generate embeddings
        embeddings = generate_t5xl_embeddings(
            fasta_file=FASTA_FILE,
            output_npy=OUTPUT_NPY,
            max_length=40
        )
        
        if embeddings is not None:
            # Verify embeddings
            verify_embeddings(OUTPUT_NPY, EXPECTED_SEQUENCES)
            
            print("\n" + "="*70)
            print("✅ SUCCESS!")
            print("="*70)
            
            print(f"\n📝 What was created:")
            print(f"   File: {OUTPUT_NPY}")
            print(f"   Shape: {embeddings.shape}")
            print(f"   Ready for MIC prediction!")
            
            print(f"\n🎯 Next Steps:")
            print(f"   1. Load genomic features for each bacterial strain")
            print(f"   2. Use {OUTPUT_NPY} in MIC prediction script")
            print(f"   3. Get MIC predictions for your Antimicrobial Peptide peptides!")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("\n💡 Common issues:")
        print("   1. Make sure FASTA file exists and path is correct")
        print("   2. Install required packages:")
        print("      pip install transformers torch biopython")
        print("   3. Ensure you have enough disk space (~2GB for model)")
        print("   4. Check that sequences are valid amino acids")

