# YOLOv8 道路标线检测学习项目

## 📚 学习路径

### 第一阶段：环境配置
#### 1. 开发环境搭建
- Python和PyCharm安装配置
- 参考：[Python+PyCharm安装教程](https://blog.csdn.net/junleon/article/details/120698578)
  
#### 2. 创建虚拟环境
- 使用Conda创建独立的Python环境
- 参考：[Conda环境管理教程](https://blog.csdn.net/weixin_45131680/article/details/141566752)

#### 3. 安装YOLOv8及相关依赖
- **安装 CUDA 与 cuDNN**：
  ```bash
  conda install nvidia::cudnn -c nvidia
- **安装PyTorch、OpenCV、Ultralytics等库**：
   ```bash
  conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia；
  （本次安装cuda13.0，cuda13.0也可兼容pytorch-cuda=12.1）
- 参考：[YOLOv8环境配置](https://blog.csdn.net/qq_67105081/article/details/137519207)

#### 4. 环境验证与测试

**PyTorch验证**
```python
import torch
import sys

print("=" * 60)
print("系统环境验证")
print("=" * 60)
print(f"Python版本: {sys.version.split()[0]}")
print(f"PyTorch版本: {torch.__version__}")
print(f"编译CUDA版本: {torch.version.cuda}")
print(f"CUDA是否可用: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"检测到的CUDA版本: 13.0")
    print(f"GPU设备: {torch.cuda.get_device_name(0)}")
    
    # 实际计算测试
    print("\n🧪 运行GPU计算测试...")
    a = torch.randn(1000, 1000).cuda()
    b = torch.randn(1000, 1000).cuda()
    c = torch.matmul(a, b)
    print(f"GPU计算测试结果: 矩阵 {c.shape} 计算成功!")
    print("✅ 环境配置成功！")
else:
    print("❌ CUDA不可用，请检查安装")
```

**测试YoloV**
```python
from ultralytics import YOLO
# 加载预训练的 YOLOv11n 模型
model = YOLO('yolo11n.pt')
# 定义图像文件的路径
source = 'cat.png' #更改为自己的图片路径
# 运行推理，并附加参数
model.predict(source, save=True)
```

### 第二阶段：数据收集准备

#### 可用数据集

- `RoadMarkingData/` — 标线数据集  
- `D-道路车道线检测1700数据集左转右转直行/` — 车道线检测数据集  

---

#### 数据准备步骤

1. **数据标注**

   - **推荐工具**：
     - [LabelImg](https://gitcode.com/gh_mirrors/lab/labelImg?utm_source=highlight_word_gitcode&word=labelimg&isLogin=1&from_link=4d701093206bac8520537a6e6a98bcb3)
     - LabelStudio
     - CVAT

   - **标注格式**：YOLO 格式（生成 `.txt` 标注文件）

   **LabelImg 标注操作步骤**：

   （1） 创建 LabelImg 的 Conda 环境并安装：
      ```bash
      pip install labelimg
      ```

   （2） 准备数据目录结构（推荐如下）：
      ```
      data/
      ├── JPEGImages/      # 存放所有图片
      ├── labels/          # 存放生成的标签文件（.txt）
      └── classes.txt      # 每行一个类别名称
      ```
   （3）开始标注，界面示例如下：
      ```bash
      conda activate labelimg
      cd [Data Path]:cd E:/;cd 'E:\Coding tools\PyCharm\labelImg-master\data'
      labelimg JPEGImages classes.txt
      ```
      ![LabelImg 标注界面](https://github.com/user-attachments/assets/d2b1750e-2425-41cd-9702-7bb85f9c56d8)

3. **数据集划分**

   - 训练集：75%  
   - 验证集：15%  
   - 测试集：15%

### 第三阶段：模型训练

#### 1. 创建数据集配置文件
- 配置data.yaml文件，定义数据集路径和类别

#### 2. 模型训练
- 使用YOLOv8进行目标检测训练
- 参考：[YOLOv8目标检测步骤](https://blog.csdn.net/qq_67105081/article/details/137545156)

### 第四阶段：实验验证

#### 1. 验证模型性能
- 计算mAP、精确率、召回率等指标

#### 2. 可视化检测效果
- 在测试集上可视化检测结果
- 分析模型表现


## 📦 数据下载

数据集已上传至百度网盘：
- **链接**: https://pan.baidu.com/s/1eKkHXpR7fFM0bVzIA95pJg?pwd=y6ga
- **提取码**: y6ga
## 🗂️ 网盘内容文件结构
**文件夹说明：**
- `RoadMarkingData/` - 标线数据集
- `D-道路车道线检测1700数据集左转右转直行/` - 车道线检测数据集
- `ultralytics-main/` - YOLOv8环境配置
- `igg_2.3.8/` - 科学上网工具
- `README.md` - 项目说明文档
## 🛠️ 技术栈

- **框架**: YOLOv8
- **深度学习库**: PyTorch
- **图像处理**: OpenCV
- **开发环境**: Python + PyCharm

## 📖 参考资料

1. [YOLOv8原理解释](https://blog.csdn.net/bingoaaa/article/details/142694776)
2. [YOLOv8环境配置教程](https://blog.csdn.net/qq_67105081/article/details/137519207)
3. [YOLOv8目标检测完整步骤](https://blog.csdn.net/qq_67105081/article/details/137545156)

## 🎯 学习目标

通过本项目，你将掌握：
- YOLOv8环境配置和模型训练
- 目标检测数据集的准备和处理
- 深度学习模型的训练和验证流程
- 道路标线检测的实际应用

## 💡 注意事项

- **安装 CUDA**  
  根据您的 GPU 版本选择对应的 CUDA 版本进行安装。
- **安装 cuDNN**  
  [NVIDIA cuDNN](https://developer.nvidia.com/cudnn) 是基于 GPU 加速的深度神经网络库。  
  **⚠️ 注意**：cuDNN 版本需与 CUDA 版本匹配。
- **配置 Python 环境**  
  建议使用 conda 创建虚拟环境，并注意 Python 版本兼容性。
- **安装 PyTorch**  
  安装时需选择与 CUDA 版本对应的 PyTorch 版本。
- **验证 YOLOv8 环境**  
  安装完成后，请运行简单测试确认环境配置成功。
---

## 📬 联系与交流
欢迎来到本项目！我非常期待与各位开发者交流想法、讨论技术细节或探索合作可能。如果您有任何疑问、建议，或只是想聊聊相关技术，随时欢迎联系我：
**📧 邮箱：** [crs9charles@foxmail.com](mailto:crs9charles@foxmail.com)
感谢您的关注与支持，希望我们能一起让这个项目变得更好！🚀

---

**⭐ 如果您觉得这个项目有帮助，欢迎点个 Star！您的支持是我持续更新的动力。**
