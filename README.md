# 修改MTLFace方法，实现DenseNet121在AIFR上的应用

通过修改AIFR中的backbone，将backbone从ResNet50修改为DenseNet121，实现了DenseNet121在AIFR上的训练。

# 环境配置方法

(假设已经安装了anaconda)

从environment.yml中创建环境

```bash
conda env create -f environment.yml
```

激活环境

```bash
conda activate mtlface
```

从requirements.txt中安装依赖

```bash
pip install -r requirements.txt
```

# 数据集准备

1. 下载CASIA-Webface（网址见README_original.md）并解压到一个指定目录

2. 分别运行以下命令（注意，命令中的文件夹目录需要根据自己的情况修改）

   ```bash
    python dataset/convert_insightface.py --source /home/kinggtree/faces_webface_112x112 --dest /home/kinggtree/casia-webface-112x112-arcface
    python dataset/convert_insightface.py --bin --source /home/kinggtree/faces_webface_112x112/agedb_30.bin --dest /home/kinggtree/arcface-test-set
   ```
  
3. 下载annotations（casia-webface.txt）（网址见README_original.md）

4. 将annotations和两个数据集文件夹移动到dataset文件夹（MTLFace文件夹中的dataset）下。其中，casia-webface-112x112-arcface 数据集文件夹需要改名为 casia-webface，casia-webface.txt 需要改名为 scaf.txt

# 训练方法

在MTLFace文件夹下运行以下命令

```bash
python main.py \
    --train_fr --backbone_name densenet --head_s 64 --head_m 0.35 \
    --weight_decay 5e-4 --momentum 0.9 --fr_age_loss_weight 0.001 --fr_da_loss_weight 0.002 --age_group 7 \
    --gamma 0.1 --milestone 20000 23000 --warmup 1000 --learning_rate 0.1 \
    --dataset_name scaf --image_size 112 --num_iter 36000 --batch_size 128 --amp
```

# 原有方法（ResNet50）的训练方法

在MTLFace文件夹下运行以下命令

```bash
python main.py \
    --train_fr --backbone_name ir50 --head_s 64 --head_m 0.35 \
    --weight_decay 5e-4 --momentum 0.9 --fr_age_loss_weight 0.001 --fr_da_loss_weight 0.002 --age_group 7 \
    --gamma 0.1 --milestone 20000 23000 --warmup 1000 --learning_rate 0.1 \
    --dataset_name scaf --image_size 112 --num_iter 36000 --batch_size 128 --amp
```
