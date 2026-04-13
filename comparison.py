import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


DATASETS = {
    "Original": "Classification_Experiments/Base_Dataset_1/results",
    "DDPM extra 5000 images added": "Classification_Experiments/Augmented_Dataset_4_2/results", 
    "LDM  extra 5000 images added": "Classification_Experiments/Augmented_Dataset_vdm_2/results",
    "DDPM VARIANCE  extra 5000 images added": "Classification_Experiments/Augmented_Dataset_ddpm_variance_V2/results"
}


MODELS = {
    "resnet18": "ResNet-18",
    "convnext_tiny": "Convnext-Tiny", 
    "swin_t": "Swin-T",
    "vit_tiny": "ViT-Tiny"
}


SAVE_DIR = "Classification_Experiments/extra_images5000_added_ddpm_vdm_variance"
os.makedirs(SAVE_DIR, exist_ok=True)

# ================= Data Extraction =================
def extract_max_accuracies():
    """
    """
    results = {}
    
    for dataset_name, dataset_path in DATASETS.items():
        dataset_results = {}
        
        for model_id, model_name in MODELS.items():
            csv_path = os.path.join(dataset_path, f"{model_id}_metrics.csv")
            
            if not os.path.exists(csv_path):
                print(f"⚠️ Warning: File not found {csv_path}")
                dataset_results[model_name] = 0
                continue
                
            try:
                df = pd.read_csv(csv_path)
                max_acc = df['test_acc'].max() * 100  # Convert to percentage
                dataset_results[model_name] = round(max_acc, 2)
                print(f"✅ {dataset_name} - {model_name}: {max_acc:.2f}%")
            except Exception as e:
                print(f"❌ Error reading {csv_path}: {e}")
                dataset_results[model_name] = 0
                
        results[dataset_name] = dataset_results
    
    return results

# ================= Bar Chart Generation =================
def create_comparison_bar_chart(results):
    """
    Create bar chart comparing five datasets
    """
    # Prepare data
    model_names = list(MODELS.values())
    dataset_names = list(DATASETS.keys())
    
    # Create DataFrame for plotting
    data_for_df = []
    for dataset_name, model_accs in results.items():
        for model_name, acc in model_accs.items():
            data_for_df.append({
                "Dataset": dataset_name,
                "Model": model_name,
                "Accuracy": acc
            })
    
    df = pd.DataFrame(data_for_df)
    
    # Create bar chart
    plt.figure(figsize=(16, 9))
    
    # Set bar positions
    x = np.arange(len(model_names))  # Model positions
    width = 0.15  # Bar width (adjusted for 5 datasets)
    
    # Plot bars for each dataset
    colors = ['#3498DB', '#2ECC71', '#E74C3C', '#9B59B6', '#F39C12']  # Blue, Green, Red, Purple, Orange
    bars = []
    
    # Calculate offset to center bars
    offset = (len(dataset_names) - 1) * width / 2
    
    for i, dataset_name in enumerate(dataset_names):
        dataset_accs = [results[dataset_name][model] for model in model_names]
        bar = plt.bar(x + i*width - offset, dataset_accs, width, 
                     label=dataset_name, color=colors[i], edgecolor='black')
        bars.append(bar)
    
    # Add decorations
    plt.xlabel('Model', fontsize=14, fontweight='bold')
    plt.ylabel('Maximum Test Accuracy (%)', fontsize=14, fontweight='bold')
    plt.title('ddpm_vdm Comparison: Maximum Test Accuracy', fontsize=16, fontweight='bold', pad=20)
    plt.xticks(x, model_names, fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.ylim(0, 110)  # Leave space for labels
    
    # Add value labels on top of bars
    for bar_group in bars:
        for bar in bar_group:
            height = bar.get_height()
            plt.annotate(f'{height}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Add grid lines
    plt.grid(True, axis='y', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    
    # Save image
    output_path = os.path.join(SAVE_DIR, "ddpm_vdm_Bar_Chart.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n📊 Bar chart saved to: {output_path}")
    
    return df

# ================= Summary Table Generation =================
def create_summary_table(results_df):
    """
    Generate summary table and save as CSV
    """
    # Pivot table for better view
    pivot_df = results_df.pivot(index='Model', columns='Dataset', values='Accuracy')
    
    # Calculate improvement percentages for all augmentations
    if 'Original' in pivot_df.columns:
        if 'DDPM extra 5000 images added' in pivot_df.columns:
            pivot_df['DDPM extra 5000 images added(%)'] = pivot_df['DDPM extra 5000 images added'] - pivot_df['Original']
        
        if 'LDM extra 5000 images added' in pivot_df.columns:
            pivot_df['DDPM extra 5000 images added(%)'] = pivot_df['LDM extra 5000 images added'] - pivot_df['Original']
        
        if 'DDPM VARIANCE  extra 5000 images added' in pivot_df.columns:
            pivot_df['DDPM VARIANCE  extra 5000 images added(%)'] = pivot_df['DDPM VARIANCE  extra 5000 images added'] - pivot_df['Original']
        
        if 'extra 5000 images added' in pivot_df.columns:
            pivot_df['extra 5000 images added(%)'] = pivot_df['extra 5000 images added'] - pivot_df['Original']
    
    # Save as CSV
    csv_path = os.path.join(SAVE_DIR, "DDPM_VARIACNE_LDM_Data_Augmentation_Comparison_Summary.csv")
    pivot_df.to_csv(csv_path)
    
    print(f"📋 Summary table saved to: {csv_path}")
    
    # Print table
    print("\n" + "="*60)
    print("DDPM_VARIACNE_LDM Data Augmentation Comparison Summary")
    print("="*60)
    print(pivot_df.to_string())
    
    return pivot_df

# ================= Main Function =================
def main():
    print("🚀 Starting five-way data augmentation comparison analysis...")
    print("="*60)
    
    # 1. Extract data
    print("📈 Extracting maximum accuracy for each model...")
    results = extract_max_accuracies()
    
    # 2. Create bar chart
    print("\n🎨 Generating comparison bar chart...")
    results_df = create_comparison_bar_chart(results)
    
    # 3. Generate summary table
    print("\n📋 Generating summary table...")
    summary_df = create_summary_table(results_df)
    
    print("\n" + "="*60)
    print("✨ Analysis completed!")
    print(f"📁 All results saved to: {SAVE_DIR}")
    print("="*60)

if __name__ == "__main__":
    main()
