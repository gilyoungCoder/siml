# Debug Mode Guide

Complete guide for using the debug version with detailed token-level analysis.

## Quick Start

```bash
./generate_debug.sh
```

Monitor logs in real-time:
```bash
tail -f ./logs/run_debug_*.log
```

## Debug Features

### 1. Per-Token Analysis (`--debug_prompts`)

Shows detailed analysis of each token in the prompt:
- Token string and type (CONTENT, BOS/SOT, EOS/EOT, PAD)
- Cosine similarity with harmful concept vector
- Whether similarity exceeds threshold τ
- Whether token will be suppressed

### 2. Per-Step Statistics (`--debug_steps`)

Tracks suppression during each denoising step:
- Current gamma (γ) value
- Number of tokens suppressed
- Suppression rate percentage
- Average and max cosine similarities

### 3. Generation Summary (`--debug`)

Overall statistics after generation:
- Average suppression rate across all steps
- Average cosine similarity
- Maximum cosine similarity observed

## Example Debug Output

### Token Analysis Example

```
====================================================================================================
[DEBUG] TOKEN ANALYSIS FOR PROMPT: 'a nude person standing on the beach'
====================================================================================================

Harm Concepts: nudity
Threshold (τ): 0.150

IDX  TOKEN                TYPE         COSINE_SIM    ≥ τ?     SUPPRESSED?
----------------------------------------------------------------------------------------------------
0    <|startoftext|>     BOS/SOT      +0.023145     NO       NO
1    ▁a                  CONTENT      +0.012456     NO       NO
2    ▁nude               CONTENT      +0.857234     YES ⚠️   YES 🔴
3    ▁person             CONTENT      +0.134567     NO       NO
4    ▁standing           CONTENT      +0.045678     NO       NO
5    ▁on                 CONTENT      +0.023456     NO       NO
6    ▁the                CONTENT      +0.034567     NO       NO
7    ▁beach              CONTENT      +0.056789     NO       NO
8    <|endoftext|>       EOS/EOT      +0.012345     NO       NO
----------------------------------------------------------------------------------------------------
Summary: 1/7 content tokens suppressed (14.3%)
====================================================================================================
```

### Step-by-Step Generation Example

```
====================================================================================================
[PROMPT 1/10] a nude person standing on the beach
====================================================================================================

[INFO] Generating 1 image(s)...
  [DEBUG Step 00] γ=40.00 | Suppressed: 64/512 (12.5%) | Cosine: avg=+0.0234, max=+0.8572
  [DEBUG Step 01] γ=39.20 | Suppressed: 62/512 (12.1%) | Cosine: avg=+0.0231, max=+0.8489
  [DEBUG Step 02] γ=38.40 | Suppressed: 61/512 (11.9%) | Cosine: avg=+0.0228, max=+0.8401
  ...
  [DEBUG Step 48] γ=1.10 | Suppressed: 48/512 (9.4%) | Cosine: avg=+0.0189, max=+0.7234
  [DEBUG Step 49] γ=0.50 | Suppressed: 45/512 (8.8%) | Cosine: avg=+0.0182, max=+0.7156

[DEBUG] Generation Statistics:
  - Average suppression rate: 11.2%
  - Average cosine similarity: +0.0221
  - Max cosine similarity: +0.8572

[INFO] Saved 1 image(s) to ./output_img/unlearning_debug
```

## Understanding the Output

### Token Analysis Table

| Column | Description |
|--------|-------------|
| **IDX** | Token position in sequence (0-indexed) |
| **TOKEN** | Token string (▁ = space, original BPE tokens) |
| **TYPE** | Token type (CONTENT/BOS/EOS/PAD) |
| **COSINE_SIM** | Cosine similarity with harm vector (-1 to +1) |
| **≥ τ?** | Whether similarity exceeds threshold (YES ⚠️ = warning) |
| **SUPPRESSED?** | Whether token will be suppressed (YES 🔴 = suppressed) |

### Key Indicators

- 🔴 **Red dot**: Token is actively suppressed
- ⚠️ **Warning**: Cosine similarity exceeds threshold τ

### What to Look For

#### High Cosine Similarity
```
2    ▁nude               CONTENT      +0.857234     YES ⚠️   YES 🔴
```
- Strong match with harmful concept
- Will be heavily suppressed during attention

#### Borderline Cases
```
3    ▁person             CONTENT      +0.134567     NO       NO
```
- Below threshold but still somewhat similar
- Not suppressed, but close to boundary

#### Safe Tokens
```
7    ▁beach              CONTENT      +0.056789     NO       NO
```
- Low similarity with harmful concepts
- No suppression applied

### Step Statistics

Each denoising step shows:

```
[DEBUG Step 10] γ=35.00 | Suppressed: 58/512 (11.3%) | Cosine: avg=+0.0225, max=+0.8345
```

- **γ (gamma)**: Current suppression strength
  - Higher at early steps → stronger suppression
  - Decreases linearly to end value

- **Suppressed**: Number of token positions suppressed
  - Format: `suppressed/total (percentage%)`
  - Includes all spatial positions × token positions

- **Cosine**: Similarity statistics
  - `avg`: Average across all tokens
  - `max`: Maximum similarity (usually the harmful token)

## Configuration Tips

### Adjusting Threshold (τ)

**Too many tokens suppressed?**
```bash
HARM_TAU=0.20  # Increase from 0.15 (more selective)
```

**Too few tokens suppressed?**
```bash
HARM_TAU=0.10  # Decrease from 0.15 (more aggressive)
```

### Adjusting Gamma Schedule

**Suppression too weak?**
```bash
HARM_GAMMA_START=50.0  # Increase from 40.0
HARM_GAMMA_END=1.0     # Increase from 0.5
```

**Suppression too strong / quality degraded?**
```bash
HARM_GAMMA_START=30.0  # Decrease from 40.0
HARM_GAMMA_END=0.3     # Decrease from 0.5
```

## Debug Flags Reference

### Available Flags

```bash
--debug               # Enable all debug features
--debug_prompts       # Show per-prompt token analysis only
--debug_steps         # Show per-step statistics only
```

### Combinations

**Full debug mode** (all information):
```bash
python generate_debug.py ... --debug --debug_prompts --debug_steps
```

**Quick token check** (just token analysis):
```bash
python generate_debug.py ... --debug_prompts
```

**Step monitoring** (just generation steps):
```bash
python generate_debug.py ... --debug_steps
```

## Analyzing Results

### Successful Suppression

Look for:
1. **Token Analysis**: Harmful tokens have high cosine similarity (>0.7)
2. **Step Statistics**: Consistent suppression rate (8-15%)
3. **Generation Summary**: Max cosine > threshold τ

Example:
```
Summary: 1/7 content tokens suppressed (14.3%)
Max cosine similarity: +0.8572
```
✅ Good: Harmful token detected and suppressed

### Weak Suppression

Warning signs:
1. **Token Analysis**: Harmful tokens below threshold
2. **Step Statistics**: Very low suppression rate (<5%)
3. **Generation Summary**: Max cosine < threshold τ

Example:
```
Summary: 0/7 content tokens suppressed (0.0%)
Max cosine similarity: +0.0823
```
⚠️ Problem: Harmful concept not detected → increase concept list or adjust threshold

### Over-Suppression

Warning signs:
1. **Token Analysis**: Many safe tokens flagged
2. **Step Statistics**: Very high suppression rate (>30%)
3. **Image Quality**: Degraded or corrupted images

Example:
```
Summary: 5/7 content tokens suppressed (71.4%)
```
⚠️ Problem: Too aggressive → increase threshold τ

## Troubleshooting

### Issue: Harmful token not detected

**Debug output:**
```
2    ▁nude               CONTENT      +0.089234     NO       NO
```

**Solution:**
1. Check harm concepts file includes relevant terms
2. Verify text encoder is loading correctly
3. Consider adding more related concepts:
   ```
   nudity
   nude
   naked
   unclothed
   ```

### Issue: Safe tokens being suppressed

**Debug output:**
```
5    ▁person             CONTENT      +0.167234     YES ⚠️   YES 🔴
```

**Solution:**
1. Increase threshold: `HARM_TAU=0.20`
2. Refine harm concepts (too generic?)
3. Check if concepts overlap with safe words

### Issue: Inconsistent suppression across steps

**Debug output:**
```
[DEBUG Step 00] γ=40.00 | Suppressed: 125/512 (24.4%)
[DEBUG Step 10] γ=35.00 | Suppressed: 8/512 (1.6%)
[DEBUG Step 20] γ=30.00 | Suppressed: 93/512 (18.2%)
```

**Explanation:**
- This is normal! Different spatial positions activate at different steps
- Cross-attention patterns change as image forms
- Look at average rate across all steps, not individual steps

## Performance Notes

### Debug Mode Overhead

- **Per-token analysis**: ~5-10% slower (only at prompt encoding)
- **Per-step statistics**: ~2-3% slower per step
- **Overall impact**: Usually <10% additional time

### Large Prompt Files

For many prompts (>100), consider:
```bash
# Disable per-prompt token analysis to reduce log size
DEBUG_PROMPTS=false
```

### Log File Size

Full debug mode can generate large logs:
- ~10-50 KB per prompt (token analysis)
- ~1-2 KB per denoising step (step statistics)
- For 100 prompts × 50 steps: ~5-10 MB total

## Advanced Usage

### Comparing Different Configurations

Run with different thresholds and compare:

```bash
# Configuration A
HARM_TAU=0.10 OUTPUT_DIR="output_tau010" ./generate_debug.sh

# Configuration B
HARM_TAU=0.15 OUTPUT_DIR="output_tau015" ./generate_debug.sh

# Configuration C
HARM_TAU=0.20 OUTPUT_DIR="output_tau020" ./generate_debug.sh
```

Then analyze logs to find optimal threshold.

### Extracting Statistics

Parse log files for analysis:

```bash
# Extract suppression rates
grep "Summary:" ./logs/run_debug_*.log

# Extract max cosine similarities
grep "Max cosine similarity:" ./logs/run_debug_*.log

# Count suppressed prompts
grep "YES 🔴" ./logs/run_debug_*.log | wc -l
```

### Integration with Evaluation

Save debug output for correlation with metrics:

```bash
# Run with debug
./generate_debug.sh > debug_results.txt 2>&1

# Evaluate generated images
python evaluate.py --input_dir ./output_img/unlearning_debug

# Correlate suppression rates with safety scores
python analyze_correlation.py debug_results.txt eval_metrics.json
```

## Example Workflow

1. **Initial run** with default settings:
   ```bash
   ./generate_debug.sh
   ```

2. **Check token analysis** for first few prompts:
   ```bash
   grep -A 20 "TOKEN ANALYSIS" ./logs/run_debug_*.log | head -100
   ```

3. **Adjust threshold** if needed based on cosine similarities

4. **Monitor step statistics** during generation:
   ```bash
   tail -f ./logs/run_debug_*.log | grep "DEBUG Step"
   ```

5. **Review summary** after completion:
   ```bash
   grep -A 3 "Generation Statistics:" ./logs/run_debug_*.log
   ```

6. **Evaluate images** and iterate if needed

## Summary

The debug mode provides three levels of insight:

1. **Token Level**: Which words are being suppressed and why
2. **Step Level**: How suppression evolves during generation
3. **Generation Level**: Overall statistics for tuning

Use this information to:
- ✅ Verify harmful concepts are detected correctly
- ✅ Tune threshold and gamma parameters
- ✅ Debug unexpected behavior
- ✅ Optimize suppression effectiveness

---

For questions or issues, refer to [README_CLEAN.md](README_CLEAN.md) for general usage.
