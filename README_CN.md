# 散户股票 Agents（Tushare/AkShare + DeepSeek/Gemini）

这是一个面向散户的轻量股票决策助手：
- 数据源：Tushare / AkShare（A股）
- 分析：多 Agent 评分（趋势 / 估值 / 资金 / 风险）
- 解释：DeepSeek / Gemini 中文建议（可选）
- 展示：Streamlit 图形界面（手机适配）

## 1. 安装

```powershell
cd D:\股票
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -r requirements.txt
```

## 2. 配置

复制 `.env.example` 为 `.env`，并填写：

```env
DATA_SOURCE=auto
TUSHARE_TOKEN=你的_tushare_token
TUSHARE_BASE_URL=http://tushare.xyz
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=你的_deepseek_key
GEMINI_API_KEY=你的_gemini_key
```

说明：
- `DATA_SOURCE`：`auto` / `tushare` / `akshare`
- `auto`：优先 Tushare，失败时自动回退 AkShare
- `tushare`：必须填写 `TUSHARE_TOKEN`
- `akshare`：可不填 `TUSHARE_TOKEN`
- `TUSHARE_BASE_URL` 可保留默认 `http://tushare.xyz`
- `LLM_PROVIDER` 可选 `deepseek` 或 `gemini`

## 3. 运行

```powershell
cd D:\股票
.\.venv\Scripts\streamlit run app.py
```

## 4. 功能

- 名称/代码搜索并选择股票（如“茅台”“600519”）
- 可切换数据源（自动 / 仅 Tushare / 仅 AkShare）
- 一键保存配置，刷新后保留 Token 和 API Key
- 输出建议（买入 / 观望 / 减仓）与仓位、止损、止盈
- 展示新闻舆情（近 7 天情绪）
- 若非交易日，自动回退到最近交易日
- 支持 DeepSeek / Gemini 中文解释

## 5. 免责声明

本工具仅用于学习和研究，不构成任何投资建议。
