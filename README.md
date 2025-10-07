# 🏥 Kallipolis医疗共和国治理系统 (Asclepion)

## 📖 项目简介

Kallipolis医疗共和国治理系统是一个基于多智能体博弈论的医院治理实时监控仿真平台，融合了深度强化学习（MADDPG）、大语言模型（LLM）、分布式控制系统和数学策略模板等前沿技术，构建了一个16维状态空间的智能医疗治理生态系统。

### 🎯 核心特性

- **🤖 多智能体协同决策**: 5个智能体（医生、实习生、患者代表、会计师、政府监管）通过博弈论进行协同决策
- **🧠 AI驱动的智能治理**: 结合MADDPG深度强化学习和LLM大语言模型的混合决策架构
- **🔧 智能体注册中心**: 统一管理智能体创建、配置和LLM服务集成，支持环境变量驱动的API密钥管理
- **🌐 多LLM提供者支持**: 支持OpenAI、Anthropic、本地模型和Mock模式，具备优雅降级机制
- **🏛️ 民主议会机制**: 基于共识算法的议会投票系统，实现透明化的医疗治理决策
- **📊 实时监控界面**: WebSocket驱动的现代化监控面板，支持16维雷达图和性能趋势分析
- **⚖️ 神圣法典规则系统**: 动态的规则管理和执行机制，确保治理决策的合规性
- **💬 智能体群聊系统**: QQ群风格的实时对话界面，支持会期管理和历史追溯

## 🏗️ 系统架构

### 多层决策架构
```
📊 Web前端界面 (实时监控面板)
     ↕️ WebSocket双向通信
🌐 WebSocket服务器 (数据推送/订阅)
     ↕️ 回调机制
🧠 KallipolisSimulator (仿真引擎)
     ↕️ 决策流水线
🏗️ 多层决策架构:
    ├── MADDPG深度强化学习层
    ├── LLM智能生成层  
    ├── 分布式控制器层
    ├── 数学策略模板层
    └── 基础模板层
```

### Agents模组架构
```
🤖 智能体注册中心 (AgentRegistry)
    ├── 🔑 环境变量API密钥管理
    ├── 🌐 多LLM提供者支持 (OpenAI/Anthropic/Local/Mock)
    ├── 🛡️ 优雅降级机制
    └── 📊 性能监控和状态管理
         ↓
👥 角色智能体层 (RoleAgents)
    ├── 👨‍⚕️ 医生智能体 (4维动作空间)
    ├── 👩‍⚕️ 实习医生智能体 (3维动作空间)
    ├── 👥 患者代表智能体 (3维动作空间)
    ├── 💼 会计智能体 (3维动作空间)
    └── 🏛️ 政府监管智能体 (3维动作空间)
         ↓
🧠 行为决策层
    ├── 📊 数学策略模型 (基于收益函数的参数化随机策略)
    ├── 🤖 LLM智能生成 (自然语言推理转数值决策)
    ├── 🎯 行为模型系统 (理性/有界理性/情感/社交/自适应)
    └── 📈 深度学习集成 (MADDPG/DQN)
```

### 16维状态空间
1. **效率指标**: 床位利用率、手术效率、诊断准确率、治疗成功率
2. **财务指标**: 收入、成本、利润率、预算执行率
3. **质量指标**: 患者满意度、医疗质量评分、安全事故率、投诉率
4. **人力指标**: 医生工作量、护士配比、培训完成率、员工满意度

## 🚀 快速开始

### 环境要求

- Python 3.8+
- Node.js 16+ (可选，用于前端开发)
- 8GB+ RAM (推荐16GB)
- 支持WebSocket的现代浏览器

### 安装依赖

```bash
# 克隆项目
git clone https://github.com/dnimo/Asclepion.git
cd Asclepion

# 安装Python依赖
pip install -r requirements.txt

# 或使用conda环境
conda create -n asclepion python=3.9
conda activate asclepion
pip install -r requirements.txt

# 安装额外的LLM依赖
pip install httpx requests
```

### 环境变量配置 (可选)

```bash
# 配置LLM API密钥 (可选，未配置时使用Mock模式)
export OPENAI_API_KEY="sk-your-openai-api-key"
export ANTHROPIC_API_KEY="your-anthropic-api-key"

# 系统配置 (可选)
export HOSPITAL_LLM_PROVIDER="openai"      # openai, anthropic, local, mock
export HOSPITAL_LLM_PRESET="openai_gpt4"   # 预设配置
export HOSPITAL_ENABLE_LLM="true"          # 是否启用LLM
export HOSPITAL_FALLBACK_MOCK="true"       # API失败时回退到mock
```

### 启动系统

```bash
# 启动WebSocket服务器和仿真引擎
python3 websocket_server.py

# 系统启动后，访问以下地址：
# 🌐 前端界面: http://localhost:8080/frontend/websocket_demo.html
# 🔌 WebSocket端点: ws://localhost:8000
# 📊 HTTP服务器: http://localhost:8080

# 测试智能体系统 (可选)
python3 env_config_example.py test        # 测试智能体注册和LLM集成
python3 env_config_example.py interactive # 交互式智能体演示
```

### 配置文件

系统配置文件位于 `config/` 目录：

- `controller_gains.yaml`: 控制器增益参数
- `holy_code_rules.yaml`: 神圣法典规则定义
- `performance_weights.yaml`: 性能权重配置
- `simulation_scenarios.yaml`: 仿真场景设置
- `system_matrices.yaml`: 系统矩阵参数

## 📱 用户界面功能

### 🎛️ 实时监控面板

**第一排横向布局**:
- ⏰ **仿真时间**: 当前仿真步数和议会倒计时
- 📈 **关键性能指标**: 整体性能、系统稳定性、危机数量、议会会议数
- 🎮 **仿真控制**: 运行状态监控和控制按钮

**第二排横向布局**:
- 🎯 **16维系统状态雷达图**: 实时显示16个维度的系统状态
- 📊 **系统性能趋势图**: 历史性能数据的时间序列分析

### 🏛️ 议会治理系统

- **当前会期状态**: 实时显示议会会议状态、倒计时和共识进度
- **历次决议记录**: 完整的议会决策历史和决议内容追溯
- **共识算法**: 基于智能体投票的民主决策机制

### 💬 智能体群聊系统

- **QQ群风格界面**: 熟悉的聊天交互体验
- **会期标签切换**: 当前会期实时对话 vs 历史记录浏览
- **智能体识别**: 不同颜色编码区分5个智能体角色
- **消息分类**: 支持议会、危机、协商、决策等不同类型对话

### ⚖️ 神圣法典规则系统

- **动态规则管理**: 实时显示激活的治理规则
- **规则执行监控**: 跟踪规则执行状态和效果
- **共识驱动的规则创建**: 当智能体达成高度共识时自动创建新规则

## 🤖 智能体角色

### 👨‍⚕️ 医生群体 (Doctors)
- **职责**: 医疗质量控制、临床决策优化
- **关注点**: 诊断准确率、治疗效果、医疗安全
- **决策风格**: 专业导向、质量优先
- **动作空间**: 4维向量 [质量改进, 资源申请, 工作负荷调整, 安全措施]
- **数值范围**: [-1.0, 1.0] (1.0=最大正向行动, 0.0=保持现状, -1.0=最大负向行动)

### 👩‍⚕️ 实习生群体 (Interns)  
- **职责**: 学习培训、基础医疗服务
- **关注点**: 培训机会、工作负荷、技能提升
- **决策风格**: 成长导向、积极学习
- **动作空间**: 3维向量 [培训需求, 工作调整, 发展计划]

### 👥 患者代表 (Patients)
- **职责**: 患者权益保护、服务质量监督
- **关注点**: 患者满意度、医疗费用、服务体验
- **决策风格**: 患者中心、成本敏感
- **动作空间**: 3维向量 [服务改善, 可及性优化, 安全关注]

### 💼 会计群体 (Accountants)
- **职责**: 财务管理、成本控制、资源分配
- **关注点**: 财务效率、成本优化、预算管理
- **决策风格**: 财务导向、效率优先
- **动作空间**: 3维向量 [成本控制, 效率提升, 预算优化]

### 🏛️ 政府监管 (Government)
- **职责**: 政策执行、合规监督、公共利益保护
- **关注点**: 政策合规、公平性、社会效益
- **决策风格**: 政策导向、公平优先
- **动作空间**: 3维向量 [监管措施, 政策调整, 协调行动]

## 📊 数据导出与分析

### 导出格式支持

- **CSV格式**: 智能体状态、时间序列数据、元数据
- **JSON格式**: 完整的仿真状态快照
- **PKL格式**: Python对象序列化存储
- **SQLite数据库**: 结构化数据查询和分析

### 数据文件结构

```
comprehensive_export/
├── simulation_[scenario]_[timestamp]_agent_[agent_name].csv
├── simulation_[scenario]_[timestamp]_metadata.csv  
├── simulation_[scenario]_[timestamp]_timeseries.csv
├── simulation_[scenario]_[timestamp].json
├── simulation_[scenario]_[timestamp].db
└── simulation_[scenario]_[timestamp].pkl
```

## 🧪 开发与测试

### 代码结构

```
Asclepion/
├── src/hospital_governance/          # 核心仿真引擎
│   ├── agents/                       # 智能体实现
│   │   ├── agent_registry.py         # 智能体注册中心 (NEW)
│   │   ├── role_agents.py            # 角色智能体核心实现
│   │   ├── llm_action_generator.py   # LLM动作生成器
│   │   ├── llm_providers.py          # LLM服务提供者 (NEW)
│   │   ├── behavior_models.py        # 行为模型系统
│   │   ├── learning_models.py        # 深度学习模型
│   │   └── interaction_engine.py     # 交互引擎
│   ├── core/                         # 核心算法
│   ├── holy_code/                    # 规则系统
│   └── simulation/                   # 仿真框架
├── frontend/                         # Web前端界面
│   └── websocket_demo.html          # 主监控界面
├── config/                           # 配置文件
├── docs/                            # 项目文档
├── examples/                        # 示例代码
└── test_export/                     # 测试数据
```

### 运行测试

```bash
# 运行单元测试
python -m pytest tests/

# 运行集成测试
python test_integration.py
python reward_state_integration_test.py  # 奖励-状态联动测试

# 运行前端功能测试
python test_frontend_fixes.py

# 智能体系统测试
python env_config_example.py test        # 测试智能体注册和LLM
python detailed_action_test.py           # 详细动作生成测试
python debug_mock.py                     # 调试Mock LLM响应

# 行为模型演示
python examples/behavior_models_demo.py
```

### 代码质量工具

```bash
# 代码格式化
black src/ tests/

# 类型检查
mypy src/

# 代码检查
flake8 src/
```

## 📈 性能优化

### 系统要求

- **最低配置**: 4GB RAM, 双核CPU
- **推荐配置**: 16GB RAM, 四核CPU, SSD存储
- **大规模仿真**: 32GB+ RAM, 多核CPU, GPU加速

### 优化建议

1. **内存优化**: 定期清理历史数据，设置合理的数据保留期
2. **计算优化**: 启用多进程并行计算，合理配置智能体数量
3. **网络优化**: 使用WebSocket连接池，优化数据传输频率
4. **存储优化**: 定期归档历史数据，使用数据库索引优化查询

## 🛠️ 配置说明

### 仿真参数调优

在 `config/simulation_scenarios.yaml` 中调整：

```yaml
scenarios:
  default:
    duration: 1000              # 仿真步数
    agent_count: 5              # 智能体数量
    parliament_interval: 7      # 议会召开间隔
    crisis_probability: 0.1     # 危机发生概率
```

### 性能权重配置

在 `config/performance_weights.yaml` 中设置各维度权重：

```yaml
weights:
  efficiency: 0.25
  financial: 0.25  
  quality: 0.25
  human_resources: 0.25
```

## 🚀 部署指南

### Docker部署

```bash
# 构建Docker镜像
docker build -t asclepion .

# 运行容器
docker run -p 8080:8080 -p 8000:8000 asclepion
```

### 生产环境部署

1. **反向代理配置** (Nginx/Apache)
2. **SSL证书配置** (HTTPS/WSS支持)
3. **负载均衡** (多实例部署)
4. **监控告警** (日志监控、性能监控)

## 📚 文档资源

- 📖 [Agents模组详细指南](docs/AGENTS_MODULE_GUIDE.md) ⭐ **NEW**
- 📖 [用户使用指南](docs/AGENTS_USAGE_GUIDE.md)
- 🧠 [行为模型指南](docs/behavior_models_guide.md)
- 🔧 [代理重构总结](docs/AGENTS_REFACTORING_SUMMARY.md)
- 📊 [数据导出总结](docs/DATA_EXPORT_COMPLETION_SUMMARY.md)
- 🎨 [布局优化总结](LAYOUT_OPTIMIZATION_SUMMARY.md)
- 🔬 [数学验证报告](docs/MATHEMATICAL_VERIFICATION_REPORT.md)

## 🤝 贡献指南

### 参与贡献

1. Fork项目仓库
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建Pull Request

### 代码规范

- 遵循PEP 8 Python代码规范
- 使用类型注解和文档字符串
- 编写单元测试覆盖新功能
- 更新相关文档

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 👨‍💻 作者

- **dnimo** - *初始开发* - [GitHub](https://github.com/dnimo)

## 🙏 致谢

- OpenAI GPT模型提供智能对话支持
- Chart.js提供数据可视化组件
- WebSocket技术实现实时通信
- 多智能体强化学习研究社区的理论支持

## 📞 支持与联系

- 🐛 **问题反馈**: [GitHub Issues](https://github.com/dnimo/Asclepion/issues)
- 💬 **讨论交流**: [GitHub Discussions](https://github.com/dnimo/Asclepion/discussions)
- 📧 **联系邮箱**: kuochingcha@gmail.com

---

**🏥 Kallipolis医疗共和国治理系统 - 让医疗治理更智能、更民主、更高效！**
