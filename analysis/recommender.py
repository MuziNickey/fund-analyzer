import os
import json
from openai import OpenAI


def get_client() -> OpenAI | None:
    """获取 AI 客户端。优先 DeepSeek，其次 Anthropic"""
    deepseek_key = os.getenv("DEEPSEEK_API_KEY", "")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")

    if deepseek_key:
        return OpenAI(api_key=deepseek_key, base_url="https://api.deepseek.com")
    if anthropic_key:
        return OpenAI(api_key=anthropic_key, base_url="https://api.anthropic.com")
    return None


def _get_model(client: OpenAI) -> str:
    """根据客户端类型返回合适的模型名"""
    if "deepseek.com" in str(client.base_url):
        return "deepseek-chat"
    return "claude-sonnet-4-6"


def _call_ai(client: OpenAI, prompt: str, max_tokens: int = 1500) -> str:
    """统一的 AI 调用接口，兼容 DeepSeek 和 Anthropic"""
    model = _get_model(client)
    resp = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content


def generate_news_analysis(client, news_text: str):
    """AI 解读新闻，返回 (sentiment_str, news_items_list)"""
    if client is None or not news_text or news_text == "暂无新闻数据":
        return "🟡中性", [{
            "title": "无法获取新闻", "source": "", "time": "",
            "summary": "请检查网络或数据源，或设置 DEEPSEEK_API_KEY",
            "sentiment": "中性", "sector": "全市场"
        }]

    prompt = f"""你是A股基金分析师。分析以下财经新闻，用JSON返回结果。

规则：
- 只选3-5条对基金投资影响最大的新闻
- sentiment 取值: "偏暖", "中性", "谨慎"
- 每条新闻的 sentiment: "利好", "利空", "中性"
- 直接返回JSON，不要有其他文字

新闻内容：
{news_text[:3000]}

JSON格式：
{{
  "sentiment": "偏暖",
  "news": [
    {{"title": "...", "source": "...", "time": "...", "summary": "...", "sentiment": "利好", "sector": "..."}}
  ]
}}"""

    try:
        text = _call_ai(client, prompt, max_tokens=1500)
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]
        data = json.loads(text.strip())
        sentiment_map = {"偏暖": "🟢偏暖", "中性": "🟡中性", "谨慎": "🔴谨慎"}
        sentiment = sentiment_map.get(data.get("sentiment", "中性"), "🟡中性")
        return sentiment, data.get("news", [])
    except Exception:
        return "🟡中性", [{
            "title": "AI 分析暂不可用", "source": "", "time": "",
            "summary": "请稍后重试", "sentiment": "中性", "sector": "全市场"
        }]


def generate_investment_advice(
    client,
    market_sentiment: str,
    top_funds_info: str,
    portfolio_diagnosis: str,
    technical_signals: str,
) -> str:
    """生成综合投资建议（Markdown格式）"""
    if client is None:
        return _fallback_advice(portfolio_diagnosis, top_funds_info)

    prompt = f"""你是资深基金投资顾问。基于以下信息给出中文投资建议。

市场情绪：{market_sentiment}

市场优质基金：
{top_funds_info[:1500]}

用户持仓诊断：
{portfolio_diagnosis[:1500]}

技术信号：
{technical_signals[:1000]}

请用Markdown格式输出：

## 持仓操作建议
| 基金 | 当前状态 | 建议 | 理由 |
（为每只持仓基金给出具体建议）

## 新基金推荐（1-2只）
每只包括：推荐理由、投资策略（定投/分批/一次性）、建议持有周期（1-3月/3-6月/6月+）、资金占比、止盈参考位、止损参考位

## 整体资金配置
持仓 vs 新投 vs 现金 的比例建议

## 风险提示"""

    try:
        return _call_ai(client, prompt, max_tokens=2500)
    except Exception as e:
        return _fallback_advice(portfolio_diagnosis, top_funds_info) + f"\n\n（AI 服务异常：{e}）"


def _fallback_advice(portfolio_diagnosis: str, top_funds_info: str) -> str:
    """无 AI 时的纯量化备用建议"""
    return f"""## 持仓操作建议（基于量化评分）

{portfolio_diagnosis}

## 市场优质基金参考

{top_funds_info}

> ⚠️ AI 分析服务未配置或不可用，以上为纯量化分析结果。
> 如需 AI 解读，请设置环境变量 `DEEPSEEK_API_KEY` 或 `ANTHROPIC_API_KEY`。

## 风险提示
本工具基于历史数据和技术指标生成分析，不构成投资建议。
历史收益不代表未来表现。基金投资有风险，入市需谨慎。"""
