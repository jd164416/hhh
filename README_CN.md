# 散户股票 Agents（Tushare + DeepSeek + 图形界面）

这是一个面向散户的轻量股票决策助手：
- 数据源：Tushare（A股）
- 分析：多 Agent 评分（趋势 / 估值 / 资金 / 风险）
- 解释：DeepSeek 中文建议（可选）
- 展示：Streamlit + Plotly 图形界面

## 1. 安装

```powershell
cd D:\股票
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -r requirements.txt
```

## 2. 配置密钥与地址

复制 `.env.example` 为 `.env`，并填写：

```env
TUSHARE_TOKEN=你的_tushare_token
TUSHARE_BASE_URL=http://tushare.xyz
DEEPSEEK_API_KEY=你的_deepseek_key
```

说明：
- `TUSHARE_TOKEN` 必填（否则无法拉 A 股数据）
- `TUSHARE_BASE_URL` 可保留默认 `http://tushare.xyz`
- `DEEPSEEK_API_KEY` 可选（不填时仍可跑规则模型）

你给的这段代码已经等价接入到项目里了：

```python
import tushare.pro.client as client
client.DataApi._DataApi__http_url = "http://tushare.xyz"
```

现在不需要手写在脚本里，只要在 `.env` 里配置 `TUSHARE_BASE_URL` 即可。

## 3. 运行

```powershell
cd D:\股票
.\.venv\Scripts\streamlit run app.py
```

## 4. 功能

- 输入股票代码（如 `600519` / `000001`）与分析日期
- 支持按股票名称搜索并选择（如“茅台”“平安”）
- 侧边栏支持“保存配置（刷新后保留）”，Token/API 无需重复输入
- 手机端自动适配（侧边栏默认收起，卡片与图表窄屏重排）
- 输出交易建议（买入 / 观望 / 减仓）
- 图形展示 K 线、均线、MACD、RSI、成交量
- 输出风险控制建议（仓位、止损、止盈）
- 结果区显示“股票名称 + 代码”
- 展示各 Agent 评分与中文解释

## 5. 免责声明

本工具仅用于学习和研究，不构成任何投资建议。
