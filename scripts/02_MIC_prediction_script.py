import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# FUNCTIONS
# ============================================================

def r_squared(y_true, y_pred):
    """Custom metric for loading MB model"""
    ss_res = K.sum(K.square(y_true - y_pred))
    ss_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - ss_res/(ss_tot + K.epsilon())

def Genomesequence_concat(Feature_array, Genome_array):
    """Concatenate genomic features with T5XL embeddings"""
    arr = np.expand_dims(Genome_array, axis=1)
    arr = np.tile(arr, (1, 1, 1))
    pad_shape = (Feature_array.shape[0], 1, Feature_array.shape[2])
    arr_padded = np.pad(arr, [(0, 0), (0, 0), (0, pad_shape[2] - arr.shape[2])], mode='constant')
    concatenated_array = np.concatenate((Feature_array, arr_padded), axis=1)
    return concatenated_array

def detect_log_transformation(train_file='data/SA_X_train_40.csv'):
    """
    Automatically detect if training data was log-transformed
    Returns: (is_log_transformed, log_type)
    """
    try:
        train = pd.read_csv(train_file)
        mic_values = train['NEW-CONCENTRATION']
       
        min_val = mic_values.min()
        max_val = mic_values.max()
        mean_val = mic_values.mean()
       
        print(f"\n📊 Training Data Analysis:")
        print(f"   File: {train_file}")
        print(f"   Min: {min_val:.3f}")
        print(f"   Max: {max_val:.3f}")
        print(f"   Mean: {mean_val:.3f}")
        print(f"   Median: {mic_values.median():.3f}")
       
        # Decision logic
        if max_val < 20 and min_val > -10:
            print(f"   ✓ Data appears LOG-TRANSFORMED")
           
            if max_val < 10:
                log_type = "ln"  # Natural log or log10
                print(f"   ✓ Likely natural log (ln) or log10")
            else:
                log_type = "log2"
                print(f"   ✓ Likely log2")
           
            return True, log_type
        else:
            print(f"   ✓ Data appears to be ACTUAL MIC values (not log-transformed)")
            return False, None
           
    except Exception as e:
        print(f"\n⚠️  Cannot load training file: {e}")
        print(f"   Assuming: LOG-TRANSFORMED with natural log")
        return True, "ln"

def inverse_log_transform(predictions_log, log_type="ln"):
    """
    Convert log-transformed predictions back to actual MIC values
   
    Parameters:
    -----------
    predictions_log : array
        Predictions in log-space
    log_type : str
        Type of log transformation ('ln', 'log10', or 'log2')
   
    Returns:
    --------
    predictions_mic : array
        Predictions in actual MIC values (μg/mL)
    """
    if log_type == "ln":
        return np.exp(predictions_log)
    elif log_type == "log10":
        return 10 ** predictions_log
    elif log_type == "log2":
        return 2 ** predictions_log
    else:
        # No transformation
        return predictions_log

# ============================================================
# MAIN SCRIPT
# ============================================================

print("="*70)
print("MIC PREDICTION FOR MORICIN PEPTIDES")
print("Fixed version with log transformation")
print("="*70)

# ============================================================
# STEP 0: DETECT LOG TRANSFORMATION
# ============================================================
print("\n" + "="*70)
print("STEP 0: ANALYZING TRAINING DATA")
print("="*70)

IS_LOG_TRANSFORMED, LOG_TYPE = detect_log_transformation()

# ============================================================
# STEP 1: LOAD DATA
# ============================================================
print("\n" + "="*70)
print("STEP 1: LOADING DATA")
print("="*70)

# Load T5XL embeddings
print("\n📁 Loading T5XL embeddings...")
try:
    # Try without allow_pickle first
    try:
        T5XL_embeddings = np.load('moricin_T5XL_40.npy')
    except ValueError:
        print("   ⚠️  File requires allow_pickle=True, loading...")
        T5XL_embeddings = np.load('moricin_T5XL_40.npy', allow_pickle=True)
        # Convert to proper numeric array if needed
        if T5XL_embeddings.dtype == 'object':
            T5XL_embeddings = np.array(T5XL_embeddings, dtype=np.float32)
   
    print(f"   ✓ Shape: {T5XL_embeddings.shape}")
    n_peptides = T5XL_embeddings.shape[0]
   
    if T5XL_embeddings.shape != (n_peptides, 40, 1024):
        print(f"   ❌ ERROR: Expected shape (n, 40, 1024), got {T5XL_embeddings.shape}")
        exit(1)
       
except FileNotFoundError:
    print(f"   ❌ ERROR: moricin_T5XL_40.npy not found!")
    exit(1)

# Load genomic features
print("\n📁 Loading bacterial genomic features...")
try:
    # Try without allow_pickle first, fallback to allow_pickle=True
    try:
        SA_genomic = np.load('SA_genomic_features.npy')
        EC_genomic = np.load('EC_genomic_features.npy')
        PA_genomic = np.load('PA_genomic_features.npy')
    except ValueError:
        print("   ⚠️  Files require allow_pickle=True, loading...")
        SA_genomic = np.load('SA_genomic_features.npy', allow_pickle=True)
        EC_genomic = np.load('EC_genomic_features.npy', allow_pickle=True)
        PA_genomic = np.load('PA_genomic_features.npy', allow_pickle=True)
       
        # Convert if object arrays
        if SA_genomic.dtype == 'object':
            SA_genomic = np.array(SA_genomic, dtype=np.float32)
        if EC_genomic.dtype == 'object':
            EC_genomic = np.array(EC_genomic, dtype=np.float32)
        if PA_genomic.dtype == 'object':
            PA_genomic = np.array(PA_genomic, dtype=np.float32)
   
    print(f"   ✓ S. aureus: {SA_genomic.shape}")
    print(f"   ✓ E. coli: {EC_genomic.shape}")
    print(f"   ✓ P. aeruginosa: {PA_genomic.shape}")
   
    # Validate shapes
    assert SA_genomic.shape == (n_peptides, 84), f"SA shape mismatch! Expected ({n_peptides}, 84)"
    assert EC_genomic.shape == (n_peptides, 84), f"EC shape mismatch! Expected ({n_peptides}, 84)"
    assert PA_genomic.shape == (n_peptides, 84), f"PA shape mismatch! Expected ({n_peptides}, 84)"
   
except FileNotFoundError as e:
    print(f"   ❌ ERROR: Genomic feature file not found: {e}")
    print(f"\n   Please run the genomic feature extraction script first!")
    exit(1)

# Load peptide info (optional)
try:
    peptide_info = pd.read_csv('moricin_truncated_40.csv')
    peptide_ids = peptide_info['Peptide_ID'].values
    peptide_seqs = peptide_info['SEQUENCE'].values
    has_info = True
    print(f"\n📁 Loaded peptide information for {len(peptide_ids)} peptides")
except:
    peptide_ids = [f"Peptide_{i+1}" for i in range(n_peptides)]
    peptide_seqs = None
    has_info = False
    print(f"\n⚠️  No peptide info file, using generic IDs")

# ============================================================
# STEP 2: PREPARE INPUT DATA
# ============================================================
print("\n" + "="*70)
print("STEP 2: PREPARING INPUT DATA")
print("="*70)

print("\n🔧 Concatenating embeddings with genomic features...")
SA_data = Genomesequence_concat(T5XL_embeddings, SA_genomic)
EC_data = Genomesequence_concat(T5XL_embeddings, EC_genomic)
PA_data = Genomesequence_concat(T5XL_embeddings, PA_genomic)

print(f"   ✓ S. aureus: {SA_data.shape}")
print(f"   ✓ E. coli: {EC_data.shape}")
print(f"   ✓ P. aeruginosa: {PA_data.shape}")

assert SA_data.shape == (n_peptides, 41, 1024), "SA concatenation failed!"

# ============================================================
# STEP 3: LOAD MODELS
# ============================================================
print("\n" + "="*70)
print("STEP 3: LOADING TRAINED MODELS")
print("="*70)

print("\n🤖 Loading models...")

model_dir = 'model_max_40'
model_files = {
    'CNN': f'{model_dir}/T5_Three_CNN_40.h5',
    'BiLSTM': f'{model_dir}/T5_Three_Bi_40.h5',
    'MB': f'{model_dir}/T5_Three_MB_40.h5'
}

# Check if models exist
for name, path in model_files.items():
    if not os.path.exists(path):
        print(f"   ❌ ERROR: {path} not found!")
        exit(1)

try:
    T5_Three_CNN = load_model(model_files['CNN'])
    print("   ✓ CNN model loaded")
   
    T5_Three_Bi = load_model(model_files['BiLSTM'])
    print("   ✓ BiLSTM model loaded")
   
    T5_Three_MB = load_model(model_files['MB'], custom_objects={'r_squared': r_squared})
    print("   ✓ Multi-Branch model loaded")
   
except Exception as e:
    print(f"\n   ❌ ERROR loading models: {e}")
    exit(1)

# ============================================================
# STEP 4: PREDICT MIC VALUES
# ============================================================
print("\n" + "="*70)
print("STEP 4: PREDICTING MIC VALUES")
print("="*70)

predictions = {}

# ─────────────────────────────────────────────────────────
# S. aureus
# ─────────────────────────────────────────────────────────
print("\n" + "─"*70)
print("🦠 Staphylococcus aureus (Gram-positive)")
print("─"*70)

print("   Running CNN...")
SA_CNN_log = T5_Three_CNN.predict([SA_data[:, :40, :], SA_data[:, 40, :][:, :84]], verbose=0)

print("   Running BiLSTM...")
SA_Bi_log = T5_Three_Bi.predict([SA_data[:, :40, :], SA_data[:, 40, :][:, :84]], verbose=0)

print("   Running Multi-Branch...")
SA_MB_log = T5_Three_MB.predict([SA_data[:, :40, :], SA_data[:, :40, :], SA_data[:, 40, :][:, :84]], verbose=0)

# Ensemble (in log-space)
SA_final_log = SA_CNN_log.reshape(-1) * 0.3 + SA_Bi_log.reshape(-1) * 0.4 + SA_MB_log.reshape(-1) * 0.3

print(f"   Log-space predictions: {SA_final_log.min():.3f} to {SA_final_log.max():.3f}")

# CRITICAL: Convert to actual MIC values
if IS_LOG_TRANSFORMED:
    SA_final = inverse_log_transform(SA_final_log, LOG_TYPE)
    SA_CNN = inverse_log_transform(SA_CNN_log.reshape(-1), LOG_TYPE)
    SA_Bi = inverse_log_transform(SA_Bi_log.reshape(-1), LOG_TYPE)
    SA_MB = inverse_log_transform(SA_MB_log.reshape(-1), LOG_TYPE)
else:
    SA_final = SA_final_log
    SA_CNN = SA_CNN_log.reshape(-1)
    SA_Bi = SA_Bi_log.reshape(-1)
    SA_MB = SA_MB_log.reshape(-1)

print(f"   ✓ MIC predictions: {SA_final.min():.3f} to {SA_final.max():.3f} μg/mL")

predictions['SA'] = pd.DataFrame({
    'Peptide_ID': peptide_ids,
    'CNN': SA_CNN,
    'BiLSTM': SA_Bi,
    'MB': SA_MB,
    'MIC_Final': SA_final
})

# ─────────────────────────────────────────────────────────
# E. coli
# ─────────────────────────────────────────────────────────
print("\n" + "─"*70)
print("🦠 Escherichia coli (Gram-negative)")
print("─"*70)

print("   Running CNN...")
EC_CNN_log = T5_Three_CNN.predict([EC_data[:, :40, :], EC_data[:, 40, :][:, :84]], verbose=0)

print("   Running BiLSTM...")
EC_Bi_log = T5_Three_Bi.predict([EC_data[:, :40, :], EC_data[:, 40, :][:, :84]], verbose=0)

print("   Running Multi-Branch...")
EC_MB_log = T5_Three_MB.predict([EC_data[:, :40, :], EC_data[:, :40, :], EC_data[:, 40, :][:, :84]], verbose=0)

EC_final_log = EC_CNN_log.reshape(-1) * 0.3 + EC_Bi_log.reshape(-1) * 0.4 + EC_MB_log.reshape(-1) * 0.3

if IS_LOG_TRANSFORMED:
    EC_final = inverse_log_transform(EC_final_log, LOG_TYPE)
    EC_CNN = inverse_log_transform(EC_CNN_log.reshape(-1), LOG_TYPE)
    EC_Bi = inverse_log_transform(EC_Bi_log.reshape(-1), LOG_TYPE)
    EC_MB = inverse_log_transform(EC_MB_log.reshape(-1), LOG_TYPE)
else:
    EC_final = EC_final_log
    EC_CNN = EC_CNN_log.reshape(-1)
    EC_Bi = EC_Bi_log.reshape(-1)
    EC_MB = EC_MB_log.reshape(-1)

print(f"   ✓ MIC predictions: {EC_final.min():.3f} to {EC_final.max():.3f} μg/mL")

predictions['EC'] = pd.DataFrame({
    'Peptide_ID': peptide_ids,
    'CNN': EC_CNN,
    'BiLSTM': EC_Bi,
    'MB': EC_MB,
    'MIC_Final': EC_final
})

# ─────────────────────────────────────────────────────────
# P. aeruginosa
# ─────────────────────────────────────────────────────────
print("\n" + "─"*70)
print("🦠 Pseudomonas aeruginosa (Gram-negative)")
print("─"*70)

print("   Running CNN...")
PA_CNN_log = T5_Three_CNN.predict([PA_data[:, :40, :], PA_data[:, 40, :][:, :84]], verbose=0)

print("   Running BiLSTM...")
PA_Bi_log = T5_Three_Bi.predict([PA_data[:, :40, :], PA_data[:, 40, :][:, :84]], verbose=0)

print("   Running Multi-Branch...")
PA_MB_log = T5_Three_MB.predict([PA_data[:, :40, :], PA_data[:, :40, :], PA_data[:, 40, :][:, :84]], verbose=0)

PA_final_log = PA_CNN_log.reshape(-1) * 0.3 + PA_Bi_log.reshape(-1) * 0.4 + PA_MB_log.reshape(-1) * 0.3

if IS_LOG_TRANSFORMED:
    PA_final = inverse_log_transform(PA_final_log, LOG_TYPE)
    PA_CNN = inverse_log_transform(PA_CNN_log.reshape(-1), LOG_TYPE)
    PA_Bi = inverse_log_transform(PA_Bi_log.reshape(-1), LOG_TYPE)
    PA_MB = inverse_log_transform(PA_MB_log.reshape(-1), LOG_TYPE)
else:
    PA_final = PA_final_log
    PA_CNN = PA_CNN_log.reshape(-1)
    PA_Bi = PA_Bi_log.reshape(-1)
    PA_MB = PA_MB_log.reshape(-1)

print(f"   ✓ MIC predictions: {PA_final.min():.3f} to {PA_final.max():.3f} μg/mL")

predictions['PA'] = pd.DataFrame({
    'Peptide_ID': peptide_ids,
    'CNN': PA_CNN,
    'BiLSTM': PA_Bi,
    'MB': PA_MB,
    'MIC_Final': PA_final
})

# ============================================================
# STEP 5: SAVE RESULTS
# ============================================================
print("\n" + "="*70)
print("STEP 5: SAVING RESULTS")
print("="*70)

# Add sequences if available
if has_info:
    for strain in predictions:
        predictions[strain].insert(1, 'Sequence', peptide_seqs)

# Save individual predictions
predictions['SA'].to_csv('predictions_S_aureus.csv', index=False)
predictions['EC'].to_csv('predictions_E_coli.csv', index=False)
predictions['PA'].to_csv('predictions_P_aeruginosa.csv', index=False)

print("\n💾 Saved individual predictions:")
print("   • predictions_S_aureus.csv")
print("   • predictions_E_coli.csv")
print("   • predictions_P_aeruginosa.csv")

# Create combined results
combined = pd.DataFrame({
    'Peptide_ID': peptide_ids,
    'SA_MIC': predictions['SA']['MIC_Final'],
    'EC_MIC': predictions['EC']['MIC_Final'],
    'PA_MIC': predictions['PA']['MIC_Final']
})

if has_info:
    combined.insert(1, 'Sequence', peptide_seqs)

combined['Average_MIC'] = combined[['SA_MIC', 'EC_MIC', 'PA_MIC']].mean(axis=1)
combined['Min_MIC'] = combined[['SA_MIC', 'EC_MIC', 'PA_MIC']].min(axis=1)
combined['Max_MIC'] = combined[['SA_MIC', 'EC_MIC', 'PA_MIC']].max(axis=1)

combined.to_csv('predictions_combined.csv', index=False)
print("   • predictions_combined.csv")

# ============================================================
# STEP 6: VALIDATION & SUMMARY
# ============================================================
print("\n" + "="*70)
print("STEP 6: RESULTS VALIDATION")
print("="*70)

# Sanity check
avg_all = combined['Average_MIC'].mean()
print(f"\n📊 Overall Statistics:")
print(f"   Average MIC (all strains): {avg_all:.3f} μg/mL")

if avg_all < 0.01 or avg_all > 10000:
    print(f"\n⚠️  WARNING: MIC values outside typical range!")
    print(f"   Expected: 0.1 - 512 μg/mL for antimicrobial peptides")
    print(f"   Got: {avg_all:.2f} μg/mL")
    print(f"\n   Possible issues:")
    print(f"   1. Log transformation type incorrect")
    print(f"   2. Genomic features not matching training data")
else:
    print(f"   ✓ MIC values in reasonable range!")

print("\n📊 MIC Statistics by Strain:")
for strain_name, full_name in [('SA', 'S. aureus'), ('EC', 'E. coli'), ('PA', 'P. aeruginosa')]:
    mic = predictions[strain_name]['MIC_Final']
    print(f"\n{full_name}:")
    print(f"   Mean: {mic.mean():.3f} μg/mL")
    print(f"   Median: {mic.median():.3f} μg/mL")
    print(f"   Range: [{mic.min():.3f}, {mic.max():.3f}] μg/mL")
    print(f"   Std Dev: {mic.std():.3f} μg/mL")

# ============================================================
# STEP 7: TOP CANDIDATES
# ============================================================
print("\n" + "="*70)
print("🏆 TOP 10 CANDIDATE PEPTIDES")
print("="*70)

print("\n📌 Best Broad-Spectrum (Lowest Average MIC):")
print("─"*70)
top10 = combined.nsmallest(10, 'Average_MIC')

if has_info:
    display_cols = ['Peptide_ID', 'Sequence', 'Average_MIC', 'SA_MIC', 'EC_MIC', 'PA_MIC']
else:
    display_cols = ['Peptide_ID', 'Average_MIC', 'SA_MIC', 'EC_MIC', 'PA_MIC']

print(top10[display_cols].to_string(index=False))

# Strain-specific top candidates
for strain, name in [('SA_MIC', 'S. aureus'), ('EC_MIC', 'E. coli'), ('PA_MIC', 'P. aeruginosa')]:
    print(f"\n📌 Best Against {name}:")
    print("─"*70)
    top5 = combined.nsmallest(5, strain)
    if has_info:
        print(top5[['Peptide_ID', 'Sequence', strain]].to_string(index=False))
    else:
        print(top5[['Peptide_ID', strain]].to_string(index=False))

# ============================================================
# FINAL MESSAGE
# ============================================================
print("\n" + "="*70)
print("✅ PREDICTION COMPLETE!")
print("="*70)

print("\n📁 Output Files:")
print("   • predictions_S_aureus.csv")
print("   • predictions_E_coli.csv")
print("   • predictions_P_aeruginosa.csv")
print("   • predictions_combined.csv")

print("\n💡 Interpretation:")
print("   • Lower MIC = Better antimicrobial activity")
print("   • MIC in μg/mL (micrograms per milliliter)")
print("   • Typical range: 0.1-512 μg/mL")
print("   • Clinical breakpoints:")
print("     - Susceptible: MIC ≤ 2-8 μg/mL")
print("     - Intermediate: MIC = 16-32 μg/mL")
print("     - Resistant: MIC ≥ 64 μg/mL")

print("\n🎯 Next Steps:")
print("   1. Review top candidates in predictions_combined.csv")
print("   2. Check for broad-spectrum vs. strain-specific activity")
print("   3. Select candidates for experimental validation")
print("   4. Consider other factors: toxicity, stability, cost")

print("\n🧪 Recommended for Lab Testing:")
print("   • Top 5-10 peptides with lowest average MIC")
print("   • Peptides with specific activity against target strain")
print("   • Peptides with balanced activity across all strains")

print("\n" + "="*70)

