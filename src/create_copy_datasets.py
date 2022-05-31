mkdir -p ~/.kaggle/ && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json;
mkdir -p /kaggle/working/datasets
kaggle datasets init -p /kaggle/working/datasets;
cp /kaggle/working/*.pt /kaggle/working/datasets;