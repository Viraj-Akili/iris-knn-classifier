"""
Main execution entry point for the Enterprise ML Experimentation Pipeline.
"""

import sys
from pathlib import Path

# Add project root to sys.path if running as a script
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config import get_default_config
from src.data.loader import load_and_validate_dataset, split_dataset
from src.models.evaluate import evaluate_champion_model
from src.models.train import ModelBenchmarkEngine
from src.visualization.plots import generate_all_diagnostic_plots


def run_pipeline() -> None:
    """Execute the end-to-end reproducible ML experimentation workflow."""
    print("=" * 65)
    print("  ENTERPRISE ML EXPERIMENTATION & EVALUATION PIPELINE")
    print("  Dataset: Fisher's Iris Benchmark | Multi-Model Tournament")
    print("=" * 65)

    # 1. Configuration
    config = get_default_config()
    print(f"\n[1] Configuration Loaded: random_seed={config.random_seed}, cv_folds={config.cv.n_splits}")

    # 2. Data Ingestion & Validation
    print("\n[2] Ingesting and Validating Dataset...")
    X, y, feature_names, target_names = load_and_validate_dataset(config)
    print(f"  Total Samples     : {len(X)}")
    print(f"  Features ({len(feature_names)})     : {', '.join(feature_names)}")
    print(f"  Target Classes ({len(target_names)}): {', '.join(target_names)}")

    # 3. Stratified Train / Test Split
    print("\n[3] Splitting Data: 80% Train / 20% Untouched Holdout Test...")
    splits = split_dataset(X, y, feature_names, target_names, config)
    print(f"  Training partition : {len(splits.X_train)} samples")
    print(f"  Holdout test set   : {len(splits.X_test)} samples (locked until final evaluation)")

    # 4. Multi-Model Benchmarking with Stratified 5-Fold Cross-Validation
    print("\n[4] Running Stratified 5-Fold CV Multi-Model Tournament...")
    engine = ModelBenchmarkEngine(config)
    benchmark_results, comparison_df, champion_result = engine.run_benchmark(
        splits.X_train, splits.y_train
    )

    print("\n" + "-" * 65)
    print("  MODEL TOURNAMENT LEADERBOARD (Ranked by 5-Fold CV Score)")
    print("-" * 65)
    for idx, row in comparison_df.iterrows():
        acc_m = row["CV Accuracy Mean"] * 100
        acc_s = row["CV Accuracy Std"] * 100
        f1_m = row["CV F1 Macro"] * 100
        print(f"  {idx + 1}. {row['Model']:<24} : CV Acc = {acc_m:.2f}% (+/- {acc_s:.2f}%) | CV Macro F1 = {f1_m:.2f}%")

    print("\n" + "-" * 65)
    print(f"  CHAMPION MODEL SELECTED: {champion_result.model_name}")
    print("  Selection Basis: Highest Stratified Cross-Validation Accuracy on Training Folds")
    print("  Best Hyperparameters:")
    for k, v in champion_result.best_params.items():
        print(f"    - {k}: {v}")
    print("-" * 65)

    # 5. Final Single Evaluation on Untouched Holdout Test Set
    print("\n[5] Evaluating Champion Model on Untouched Holdout Test Set...")
    final_metrics, error_analysis, pred_df = evaluate_champion_model(
        champion_result=champion_result,
        splits=splits,
        config=config,
    )

    print("\n" + "=" * 65)
    print(f"  FINAL HOLDOUT TEST PERFORMANCE ({champion_result.model_name})")
    print("=" * 65)
    print(f"  Test Accuracy         : {final_metrics['test_accuracy'] * 100:.2f}%")
    print(f"  Test Precision (Macro): {final_metrics['test_precision_macro']:.4f}")
    print(f"  Test Recall (Macro)   : {final_metrics['test_recall_macro']:.4f}")
    print(f"  Test F1 Score (Macro) : {final_metrics['test_f1_macro']:.4f}")
    print(f"  Test F1 (Weighted)    : {final_metrics['test_f1_weighted']:.4f}")
    print(f"  Test Samples Evaluated: {final_metrics['test_sample_count']}")
    print(f"  Correct Classifications: {error_analysis.correct_count} / {error_analysis.total_test_samples}")

    # 6. Error Analysis
    print("\n" + "-" * 65)
    print("  DEEP ERROR ANALYSIS ON HOLDOUT TEST PARTITION")
    print("-" * 65)
    if error_analysis.misclassified_count == 0:
        print("  Zero misclassifications on holdout test set (100% accuracy)!")
    else:
        print(f"  Total Misclassified Samples: {error_analysis.misclassified_count}")
        for i, sample in enumerate(error_analysis.misclassified_samples, 1):
            print(f"\n  [Error #{i}] Original Row Index: {sample.sample_index}")
            print(f"    - True Class       : {sample.actual_label.upper()}")
            print(f"    - Predicted Class  : {sample.predicted_label.upper()} (Confidence: {sample.confidence * 100:.2f}%)")
            print(f"    - Class Probas     : {sample.probabilities}")
            print(f"    - Input Features   : {sample.features}")
            if sample.nearest_neighbors_context:
                print("    - 5 Nearest Training Neighbors:")
                for n_info in sample.nearest_neighbors_context:
                    if "error" in n_info:
                        print(f"        {n_info['error']}")
                    else:
                        print(f"        Row #{n_info['train_sample_id']:<3} | Class: {n_info['true_class']:<10} | Distance: {n_info['distance']:.4f}")

    # 7. Diagnostic Visualizations Generation
    print("\n[6] Generating High-Resolution Diagnostic Plots (Headless)...")
    plot_paths = generate_all_diagnostic_plots(
        comparison_df=comparison_df,
        final_metrics=final_metrics,
        error_analysis=error_analysis,
        splits=splits,
        pred_df=pred_df,
        config=config,
    )
    for plot_name, path in plot_paths.items():
        print(f"  Saved plot [{plot_name}] -> {path}")

    print("\n[7] Persisting Machine-Readable Artifacts...")
    print(f"  - Metrics: {config.paths.metrics_dir / 'final_metrics.json'}")
    print(f"  - Report : {config.paths.metrics_dir / 'classification_report.json'}")
    print(f"  - Table  : {config.paths.experiments_dir / 'model_comparison.csv'}")
    print(f"  - CV Data: {config.paths.experiments_dir / 'cv_results.csv'}")
    print(f"  - Preds  : {config.paths.predictions_dir / 'test_predictions.csv'}")
    print(f"  - Model  : {config.paths.models_dir / 'champion_pipeline.joblib'}")

    print("\n" + "=" * 65)
    print("  PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
    print("=" * 65)


if __name__ == "__main__":
    run_pipeline()
