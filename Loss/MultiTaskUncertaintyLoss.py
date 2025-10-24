import torch
import torch.nn as nn
import numpy as np

# -------------------------------------------------------------
# 1. 定义包含同方差不确定性参数的模型
# -------------------------------------------------------------
class MultiTaskModelWithUncertainty(nn.Module):
    def __init__(self):
        super(MultiTaskModelWithUncertainty, self).__init__()
        # 共享的特征提取层
        self.shared_layers = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        # 任务1的输出头 (回归)
        self.task1_head = nn.Linear(64, 1)
        # 任务2的输出头 (回归)
        self.task2_head = nn.Linear(64, 1)

        # 关键部分：为每个任务定义一个可学习的对数方差 (log(sigma^2))
        # nn.Parameter 会将这个张量注册为模型的一部分，使其在 backward() 时能接收梯度
        # 初始化为0，意味着初始权重为1 (exp(0)=1)，对两个任务一视同仁
        self.log_var_task1 = nn.Parameter(torch.zeros(1))
        self.log_var_task2 = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        shared_features = self.shared_layers(x)
        output1 = self.task1_head(shared_features)
        output2 = self.task2_head(shared_features)
        return output1, output2

# -------------------------------------------------------------
# 2. 定义自动加权的损失函数
# -------------------------------------------------------------
class MultiTaskUncertaintyLoss(nn.Module):
    def __init__(self, num_tasks=2):
        super(MultiTaskUncertaintyLoss, self).__init__()
        self.num_tasks = num_tasks

    def forward(self, pred1, target1, pred2, target2, log_var1, log_var2):
        # 定义每个任务的基础损失，这里使用均方误差(MSE)
        mse_loss = nn.MSELoss()
        loss1 = mse_loss(pred1, target1)
        loss2 = mse_loss(pred2, target2)

        # 根据公式计算加权损失
        # precision_1 = 1 / (2 * sigma_1^2) = torch.exp(-log_var1)
        # precision_2 = 1 / (2 * sigma_2^2) = torch.exp(-log_var2)
        # 为了数值稳定性，我们使用 0.5 * exp(-s) * L + 0.5 * s 的形式
        
        weighted_loss1 = 0.5 * torch.exp(-log_var1) * loss1 + 0.5 * log_var1
        weighted_loss2 = 0.5 * torch.exp(-log_var2) * loss2 + 0.5 * log_var2
        
        total_loss = weighted_loss1 + weighted_loss2
        
        # 返回总损失和各个任务的独立损失，方便监控
        return total_loss, loss1, loss2


# -------------------------------------------------------------
# 3. 创建合成数据集
# -------------------------------------------------------------
# 任务1: y1 = 2x + 1 (噪声很小，确定性高)
# 任务2: y2 = -x + 5 (噪声很大，不确定性高)
X = torch.randn(200, 1) * 10
y1_target = 2 * X + 1 + torch.randn(200, 1) * 0.5  # 小噪声
y2_target = -1 * X + 5 + torch.randn(200, 1) * 5.0 # 大噪声

# -------------------------------------------------------------
# 4. 训练过程
# -------------------------------------------------------------
model = MultiTaskModelWithUncertainty()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = MultiTaskUncertaintyLoss()

epochs = 2000

for epoch in range(epochs):
    optimizer.zero_grad()

    # 模型前向传播
    y1_pred, y2_pred = model(X)

    # 计算总损失
    total_loss, task1_loss, task2_loss = loss_fn(
        y1_pred, y1_target,
        y2_pred, y2_target,
        model.log_var_task1,
        model.log_var_task2
    )

    # 反向传播和优化
    total_loss.backward()
    optimizer.step()

    if (epoch + 1) % 200 == 0:
        # 从对数方差计算权重 (weight = exp(-log_var))
        weight1 = torch.exp(-model.log_var_task1).item()
        weight2 = torch.exp(-model.log_var_task2).item()
        
        print(f"Epoch [{epoch+1}/{epochs}] | Total Loss: {total_loss.item():.4f} | "
              f"Task1 Loss: {task1_loss.item():.4f} | Task2 Loss: {task2_loss.item():.4f}")
        print(f"  -> Uncertainty (log_var): Task1={model.log_var_task1.item():.4f}, Task2={model.log_var_task2.item():.4f}")
        print(f"  -> Learned Weights:       Task1={weight1:.4f}, Task2={weight2:.4f}\n")

# -------------------------------------------------------------
# 5. 分析结果
# -------------------------------------------------------------
print("="*20 + " 训练完成 " + "="*20)
final_log_var1 = model.log_var_task1.item()
final_log_var2 = model.log_var_task2.item()
final_weight1 = np.exp(-final_log_var1)
final_weight2 = np.exp(-final_log_var2)

print(f"最终对数方差 (log_var):")
print(f"  任务1 (低噪声): {final_log_var1:.4f}")
print(f"  任务2 (高噪声): {final_log_var2:.4f}")
print("\n最终学到的权重 (exp(-log_var)):")
print(f"  任务1 (低噪声): {final_weight1:.4f}")
print(f"  任务2 (高噪声): {final_weight2:.4f}")

if final_weight1 > final_weight2:
    print("\n结论：模型成功地为噪声更低的任务1分配了更高的权重！")
else:
    print("\n结论：模型未能按预期分配权重。")