# event_data_extractor.py
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from collections import defaultdict

def extract_tensorboard_data(event_file, output_dir=None):
    """从TensorBoard事件文件中提取所有指标数据"""
    if output_dir is None:
        output_dir = os.path.dirname(event_file)
    
    scalar_data = defaultdict(list)
    histogram_data = defaultdict(list)
    
    try:
        for event in tf.compat.v1.train.summary_iterator(event_file):
            for value in event.summary.value:
                # 标量数据提取
                if hasattr(value, 'simple_value'):
                    scalar_data[value.tag].append((event.step, value.simple_value, event.wall_time))
                # 直方图数据提取(需要更复杂的解析)
                elif hasattr(value, 'tensor') and hasattr(value, 'tag') and 'histogram' in value.metadata.plugin_data.plugin_name.lower():
                    histogram_values = tf.make_ndarray(value.tensor)
                    histogram_data[value.tag].append((event.step, histogram_values, event.wall_time))
    except Exception as e:
        print(f"提取事件数据时发生错误: {e}")
    
    print(f"发现{len(scalar_data)}个标量指标和{len(histogram_data)}个直方图指标")
    
    # 创建数据目录
    data_dir = os.path.join(output_dir, "extracted_tensorboard_data")
    os.makedirs(data_dir, exist_ok=True)
    
    # 可视化和保存标量数据
    for tag, values in scalar_data.items():
        steps, data_values, _ = zip(*values)
        
        plt.figure(figsize=(10, 6))
        plt.plot(steps, data_values)
        plt.title(f'指标: {tag}')
        plt.xlabel('步数')
        plt.ylabel('值')
        plt.grid(True, alpha=0.3)
        
        # 保存图表
        tag_filename = tag.replace('/', '_').replace(' ', '_')
        plt.savefig(os.path.join(data_dir, f"{tag_filename}.png"))
        plt.close()
        
        # 保存原始数据
        with open(os.path.join(data_dir, f"{tag_filename}.csv"), 'w') as f:
            f.write("step,value,timestamp\n")
            for step, value, timestamp in values:
                f.write(f"{step},{value},{timestamp}\n")
    
    print(f"所有数据已提取并保存至 {data_dir}")
    return scalar_data, histogram_data

# 使用示例
if __name__ == "__main__":
    event_file = "D:/fromtlx/maddpg_pe_1_rvo_v3/results/simple_tag_v3/20250228-174546/avoid_test1/tensorboard_logs/events.out.tfevents.1740735946.550omen.45788.0"
    scalar_data, histogram_data = extract_tensorboard_data(event_file)