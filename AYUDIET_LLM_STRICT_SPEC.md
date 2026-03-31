# AYUDIET STRICT LLM SPECIFICATION

# Complete Production-Ready Specification

## 🔒 CORE PRINCIPLE

**The LLM is NOT the decision maker. The LLM is a controlled signal processor.**

This document defines MANDATORY constraints for LLM integration into AYUDIET - a deterministic, schema-bound backend system for Ayurvedic nutrition assessment.

---

## 1. GLOBAL OUTPUT RULES (MANDATORY)

Apply to **ALL** LLM prompts without exception:

### ✅ REQUIRED BEHAVIORS

- Return ONLY JSON when structured output required
- NO natural language unless explicitly requested
- NO markdown (including bold, headers, lists)
- NO explanations outside defined schema
- NO extra fields beyond schema definition
- NO assumptions beyond input data
- If unsure → return LOW confidence OR minimal valid output

### ❌ FORBIDDEN BEHAVIORS

- Breaking JSON format with explanatory text
- Adding fields not in schema
- Hallucinating medical information
- Mixing structured and unstructured output
- Markdown formatting in JSON values
- Assumptions about missing data

---

## 2. SCHEMA COMPLIANCE (MANDATORY)

### 2.1 AIProfileOutput_v1 (Profiling Task)

```json
{
  "risk_flags": ["enum_values_only"],
  "dosha_estimate": {
    "vata": 0.0-1.0,
    "pitta": 0.0-1.0,
    "kapha": 0.0-1.0
  },
  "primary_dosha": "vata|pitta|kapha",
  "confidence": 0.0-1.0
}
```

**Validation Rules:**

- `risk_flags`: ONLY values from enum
  - `high_blood_pressure`, `diabetes`, `kidney_disease`, `liver_disease`
  - `heart_condition`, `thyroid_disorder`, `allergy`, `intolerance`, `none`
- `dosha_estimate.sum()`: must be 0.95-1.05 (normalized)
- `confidence`: float, 0.0-1.0
- **NO extra fields**

### 2.2 StructuredFeedback_v1 (Feedback Task)

```json
{
  "feedback_type": "LIKE|DISLIKE|REPLACE|UNCLEAR",
  "target": "string_max_500_chars"
}
```

**Validation Rules:**

- `feedback_type`: ONLY exact enum values
- If ambiguous → `UNCLEAR` (never hallucinate intent)
- `target`: brief identifier (max 500 chars)
- **NO explanations**

### 2.3 StructuredExplanation_v1 (RAG Task)

```json
{
  "explanation": "string_max_2000_chars",
  "sources": ["source_list"],
  "confidence": 0.0-1.0
}
```

**Validation Rules:**

- `explanation`: **ONLY** from provided context
- `sources`: **MUST** cite actual sources
- No fabricated citations
- No ungrounded medical claims
- Check: `has_ungrounded_claims(explanation) == False`
- Check: `has_fabricated_citations(sources) == False`

---

## 3. CONFIDENCE CALIBRATION (MANDATORY)

Confidence scores **MUST** be calibrated based on input quality, NOT just LLM confidence.

### 3.1 Calibration Thresholds

| Confidence | Decision | Behavior                      |
| ---------- | -------- | ----------------------------- |
| > 0.7      | HIGH     | Process normally, full output |
| 0.45-0.7   | MEDIUM   | Process with warnings         |
| 0.25-0.45  | LOW      | Consider fallback             |
| < 0.25     | VERY_LOW | Use fallback output           |

### 3.2 Input Signal Analysis (Deterministic)

Use `ConfidenceCalibrator` to assess input BEFORE calling LLM:

**CLEAR signals (confidence multiplier: +0.3)**

- Specific medical terms: "high blood pressure", "diabetes", "acidity"
- Quantifiable data: age, weight, symptom frequency
- Multiple correlated symptoms

**VAGUE signals (confidence multiplier: -0.15)**

- Soft language: "feel off", "sometimes", "maybe"
- Unclear references: "it", "that thing"
- Single vague complaint

**INSUFFICIENT signals**

- Empty input → return fallback (0.0 confidence)
- < 10 words → manual review
- Gibberish → fallback (< 0.3 confidence)

### 3.3 Dosha Assessment Confidence

```python
# Deterministic scoring
vata_confidence = DoshaConfidenceCalibrator.assess_dosha_confidence(
    has_age=bool,
    has_weight=bool,
    has_height=bool,
    has_symptoms=bool,
    symptom_clarity=float(0-1),
    data_consistency=bool
)
```

- Clear age+weight+symptoms → confidence 0.7+
- Partial data (e.g., age only) → confidence 0.4-0.6
- Minimal data (symptoms only) → confidence < 0.4

---

## 4. HALLUCINATION CONTROL (MANDATORY)

### 4.1 Fabricated Medical Claims

**FORBIDDEN PHRASES** (auto-flagged):

- "proven to cure"
- "guaranteed"
- "always works"
- "100% effective"
- "miracle"
- "permanently cured"

**Detection Method:**

```python
if HallucinationDetector.has_ungrounded_claims(explanation):
    reduce_confidence(0.5x)
    flag_for_review()
```

### 4.2 Fabricated Citations

**RED FLAGS:**

- Chapter numbers > 50 (unrealistic)
- "Personal communication"
- Sources not in Ayurvedic canon
- Vague attribution

**Valid Sources:**

- Charaka Samhita
- Sushruta Samhita
- Bhava Prakasha
- Dhanvantari Nighantu
- Ashtanga Hridaya

### 4.3 Context Adherence (RAG Only)

For RAG explanation task:

1. Extract key terms from explanation
2. Check if terms appear in context
3. Flag if < 50% adherence
4. Reduce confidence by 0.5x if issues detected

---

## 5. ERROR HANDLING (SAFE DEFAULTS)

### 5.1 Fallback Outputs

When LLM cannot comply, return **minimal valid outputs**:

**Profiling Fallback:**

```json
{
  "risk_flags": ["none"],
  "dosha_estimate": { "vata": 0.33, "pitta": 0.33, "kapha": 0.34 },
  "primary_dosha": "pitta",
  "confidence": 0.2
}
```

**Feedback Fallback:**

```json
{
  "feedback_type": "UNCLEAR",
  "target": "unknown"
}
```

**Explanation Fallback:**

```json
{
  "explanation": "Insufficient context to provide reliable information",
  "sources": [],
  "confidence": 0.1
}
```

### 5.2 Error Decision Tree

```
Input → Validate & Sanitize
         ↓
         Calibrate Confidence
         ↓
         [confidence < 0.25?] → Use Fallback
         ↓ (NO)
         Call LLM
         ↓
         Parse JSON
         ↓
         [valid JSON?] → NO → Use Fallback
         ↓ (YES)
         Validate Schema
         ↓
         [schema valid?] → NO → Use Fallback
         ↓ (YES)
         Check Hallucinations
         ↓
         [hallucinations?] → Reduce Confidence, Flag
         ↓
         Return Output
```

---

## 6. VALIDATION PIPELINE (MANDATORY)

All LLM outputs go through this pipeline:

### Step 1: JSON Parsing

```python
try:
    raw_data = json.loads(llm_output)
except JSONDecodeError:
    return create_fallback_output("profiling", 0.1)
```

### Step 2: Schema Validation

```python
is_valid, parsed_obj, error = OutputValidator.validate_profile_output(raw_data)
if not is_valid:
    return create_fallback_output("profiling", 0.2)
```

### Step 3: Hallucination Detection

```python
is_valid, obj, error = OutputValidator.validate_explanation_output(raw_data)
if HallucinationDetector.has_fabricated_citations(obj.sources):
    obj.confidence *= 0.5
    flag_for_review()
```

### Step 4: Confidence Adjustment

```python
final_confidence = ConfidenceCalibrator.calibrate_confidence(
    llm_confidence,
    ambiguity_analysis
)
```

---

## 7. ANTI-PATTERNS (STRICTLY FORBIDDEN)

### Diet Generation

- ❌ Do NOT generate full diet plans during profiling
- ❌ Do NOT suggest foods in assessment phase
- ❌ Do NOT override system logic with LLM suggestions
- ❌ Do NOT produce long explanations in profiling

### Output Format

- ❌ Do NOT mix JSON + markdown
- ❌ Do NOT include explanatory text outside schema
- ❌ Do NOT use undefined enum values
- ❌ Do NOT add fields beyond schema

### Hallucination

- ❌ Do NOT infer beyond input
- ❌ Do NOT fabricate medical facts
- ❌ Do NOT guess citations
- ❌ Do NOT make assumptions when uncertain

---

## 8. IMPLEMENTATION CHECKLIST

### Required Modules

- [ ] `llm_schemas_strict.py` - Schema definitions
- [ ] `llm_prompts_strict.py` - Strict prompt templates
- [ ] `llm_confidence_calibration.py` - Confidence scoring
- [ ] `llm_hallucination_control.py` - Hallucination detection
- [ ] `llm_strict_wrapper.py` - LLM wrapper (enforces all rules)
- [ ] `test_llm_strict.py` - Test suite (MANDATORY)

### Agent Updates Required

- [ ] `profile_agent` - Use `StrictLLMWrapper.extract_health_profile()`
- [ ] `knowledge_agent` - Use `StrictLLMWrapper.generate_explanation()`
- [ ] `feedback_agent` - Use `StrictLLMWrapper.parse_feedback()`
- [ ] `diet_plan_agent` - Use deterministic rules (minimal LLM use)
- [ ] `recipe_agent` - Retrieve from DB, minimal LLM generation

### Testing Requirements

- [ ] Clear input (high confidence) test passes
- [ ] Vague input (low confidence) test passes
- [ ] Invalid input (fallback) test passes
- [ ] Schema validation tests pass
- [ ] Hallucination detection tests pass
- [ ] Confidence calibration tests pass
- [ ] Integration tests pass

---

## 9. USAGE EXAMPLE

```python
from llm_strict_wrapper import StrictLLMWrapper
from agents_enhanced import get_llm

# Initialize wrapper with LLM
llm = get_llm()
wrapper = StrictLLMWrapper(llm)

# Profile extraction (STRICT schema compliance)
result = wrapper.extract_health_profile(
    user_input="I'm 28 years old, weigh 65kg, have high blood pressure",
    current_profile={}
)

# GUARANTEED:
# - Returns AIProfileOutput_v1 schema
# - Confidence is calibrated (not just LLM's)
# - Hallucinations are detected & prevented
# - Falls back to safe defaults on error

if result["success"]:
    print(f"Profile: {result['data']}")
    print(f"Confidence: {result['confidence']}")
else:
    print(f"Using fallback: {result['data']}")
```

---

## 10. DEPLOYMENT REQUIREMENTS

### Pre-Production Checklist

- [ ] All test cases pass (see `test_llm_strict.py`)
- [ ] Confidence calibration validated on real data
- [ ] Fallback mechanisms tested
- [ ] Hallucination detection reviewed by domain expert
- [ ] Schema compliance verified for all tasks
- [ ] Error rates measured and acceptable

### Monitoring

- [ ] Track confidence scores in production
- [ ] Flag outputs with confidence < 0.5
- [ ] Monitor hallucination detection hits
- [ ] Log all fallback usage
- [ ] Review low-confidence clusters

### Maintenance

- [ ] Update risk flag enum if new conditions added
- [ ] Recalibrate confidence thresholds with real data
- [ ] Add new valid sources to citations validator
- [ ] Test new model updates against all schemas

---

## 11. COMPLIANCE CERTIFICATION

This system is certified STRICT when:

✅ **All modules integrated and tested**
✅ **100% schema compliance on test suite**
✅ **Hallucination rate < 0.1%**
✅ **Confidence calibration validated**
✅ **Fallback mechanisms active**
✅ **No natural language in JSON outputs**
✅ **All anti-patterns detected & prevented**
✅ **Error handling tested**

---

## Document Version

- Version: 1.0 (AYUDIET Strict LLM Specification)
- Date: April 2026
- Status: PRODUCTION READY
- Review Cycle: Quarterly

**This specification is MANDATORY for AYUDIET LLM integration.**
