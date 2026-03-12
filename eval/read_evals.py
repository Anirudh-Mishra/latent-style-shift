import pandas as pd

# Load evaluator output
df = pd.read_csv("infedit_eval.csv")

# Paper values from InfEdit (VI* UAC)
paper_metrics = {
    "CLIP Whole": 25.03,
    "CLIP Edited": 22.22,
    "LPIPS ×10^3": 47.58,
    "MSE ×10^4": 32.09,
    "SSIM ×10^2": 85.66,
    "PSNR": 28.51,
}

# Your reproduced values from evaluator CSV
my_metrics = {
    "CLIP Whole": df["infedit|clip_similarity_target_image"].mean(),
    "CLIP Edited": df["infedit|clip_similarity_target_image_edit_part"].mean(),
    "LPIPS ×10^3": df["infedit|lpips_unedit_part"].mean() * 1000,
    "MSE ×10^4": df["infedit|mse_unedit_part"].mean() * 10000,
    "SSIM ×10^2": df["infedit|ssim_unedit_part"].mean() * 100,
    "PSNR": df["infedit|psnr_unedit_part"].mean(),
}

# Build comparison table
comparison_df = pd.DataFrame({
    "Metric": list(paper_metrics.keys()),
    "InfEdit Paper (VI* UAC)": [paper_metrics[k] for k in paper_metrics],
    "My Reproduction": [my_metrics[k] for k in paper_metrics],
})

# Optional: difference column
comparison_df["Difference (Mine - Paper)"] = (
    comparison_df["My Reproduction"] - comparison_df["InfEdit Paper (VI* UAC)"]
)

# Round for neat display
comparison_df = comparison_df.round(2)

print("\n=== InfEdit Paper vs My Reproduction ===\n")
print(comparison_df.to_string(index=False))

comparison_df.to_csv("infedit_paper_vs_mine.csv", index=False)
print("\nSaved table to infedit_paper_vs_mine.csv")