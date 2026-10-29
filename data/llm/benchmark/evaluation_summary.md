# Tomato AI Social Listening - Automated Evaluation Report

**Date & Time**: 2026-10-28 23:04:50
**Evaluation Engine**: Groq API (`openai/gpt-oss-120b`)
**Dataset**: Tomato Synthetic Social Media Dataset (`50` posts)

> [!NOTE]
> **Dataset Integrity Notice**: Reference labels in this evaluation are synthetic draft labels designed for automated testing. They represent benchmark baseline classifications and require human verification before operational deployment.

## 1. Executive Summary

- **Total Posts Evaluated**: `50`
- **JSON Schema Validity**: `50/50` (100.0%)
- **API Failures / Errors**: `0`
- **Average Latency**: `6377.11 ms` (Min: `599.52 ms`, Max: `13748.2 ms`, P95: `12552.22 ms`)
- **Overall Safety Risk Recall**: `93.33%` (14/15 critical risks captured)

## 2. Field-by-Field Evaluation Metrics

| Dimension | Accuracy | Macro Precision | Macro Recall | Macro F1 | Total Samples | Errors |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **sentiment** | 100.0% | 100.0% | 100.0% | 100.0% | 50 | 0 |
| **topic** | 78.0% | 79.6% | 75.4% | 75.1% | 50 | 11 |
| **urgency** | 84.0% | 80.5% | 82.1% | 80.3% | 50 | 8 |
| **needs_reply** | 98.0% | 100.0% | 97.5% | 98.7% | 50 | 1 |
| **risk_flag** | 96.0% | 96.6% | 92.8% | 93.9% | 50 | 2 |
| **language** | 98.0% | 75.0% | 62.5% | 66.7% | 50 | 1 |
| **sarcasm** | 100.0% | 100.0% | 100.0% | 100.0% | 50 | 0 |
| **human_review** | 96.0% | 93.3% | 93.3% | 93.3% | 50 | 2 |

## 3. High-Priority Safety & Legal Risk Audit

| Risk Category | Synthetic Reference Count | Correctly Detected | Missed as 'None' | Recall Rate |
|:---|:---:|:---:|:---:|:---:|
| **Food Safety** | 6 | 6 | 0 | 100.0% |
| **Privacy** | 3 | 3 | 0 | 100.0% |
| **Harassment** | 3 | 3 | 0 | 100.0% |
| **Legal** | 3 | 2 | 1 | 66.7% |

> [!WARNING]
> **Safety Alert**: The classifier missed one or more critical safety risks by classifying them as 'none'. Review `error_analysis.csv` immediately.

## 4. Confusion Matrices

### Sentiment Matrix (Row = Reference, Col = Predicted)

| Reference \ Pred | negative | neutral | positive |
|:---|:---:|:---:|:---:|
| **negative** | 30 | 0 | 0 |
| **neutral** | 0 | 9 | 0 |
| **positive** | 0 | 0 | 11 |

### Urgency Matrix (Row = Reference, Col = Predicted)

| Reference \ Pred | low | medium | high | critical |
|:---|:---:|:---:|:---:|:---:|
| **low** | 18 | 0 | 0 | 0 |
| **medium** | 1 | 5 | 0 | 0 |
| **high** | 0 | 3 | 8 | 1 |
| **critical** | 0 | 1 | 2 | 11 |

### Risk Flag Matrix (Row = Reference, Col = Predicted)

| Reference \ Pred | none | food_safety | privacy | harassment | legal |
|:---|:---:|:---:|:---:|:---:|:---:|
| **none** | 34 | 1 | 0 | 0 | 0 |
| **food_safety** | 0 | 6 | 0 | 0 | 0 |
| **privacy** | 0 | 0 | 3 | 0 | 0 |
| **harassment** | 0 | 0 | 0 | 3 | 0 |
| **legal** | 1 | 0 | 0 | 0 | 2 |

## 5. Artifact Links

- Predictions: [groq_predictions.csv](file:///c:/Users/ishan/Downloads/Handson%20Demo/Handson%20Demo/AIAgent/groq_predictions.csv)
- Comparison Results: [comparison_results.csv](file:///c:/Users/ishan/Downloads/Handson%20Demo/Handson%20Demo/AIAgent/comparison_results.csv)
- Field Metrics: [field_metrics.csv](file:///c:/Users/ishan/Downloads/Handson%20Demo/Handson%20Demo/AIAgent/field_metrics.csv)
- Error Analysis: [error_analysis.csv](file:///c:/Users/ishan/Downloads/Handson%20Demo/Handson%20Demo/AIAgent/error_analysis.csv)

