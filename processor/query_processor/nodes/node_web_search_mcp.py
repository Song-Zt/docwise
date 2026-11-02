import asyncio
import json

from agents.mcp import MCPServerStreamableHttp

from config.bailian_mcp_config import mcp_config
from processor.query_processor.base import NodeBase
from processor.query_processor.state import QueryGraphState
from tool.logger import logger
from utils.json_format_utils import serialize_json


class NodeWebSearchMcp(NodeBase):
    """
    节点功能，调用外部搜索引擎补充信息
    """

    # 覆盖基类的 name 属性，标识节点名称
    name: str = "node_web_search_mcp"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        """
        联网搜索是可选增强：无论无结果还是调用失败，都返回空列表，
        让后续流程继续使用本地知识库结果。
        """
        query = state.get("rewritten_query", "")
        docs = []

        if query:
            try:
                result = asyncio.run(self._mcp_call(query))
                if result and result.content:
                    pages = json.loads(result.content[0].text).get("pages") or []
                    # 统一输出结构化结果，供后续 rerank/引用使用
                    # 每条：{title, url, snippet}
                    for item in pages:
                        snippet = (item.get("snippet") or "").strip()
                        url = (item.get("url") or "").strip()
                        title = (item.get("title") or "").strip()
                        if snippet:
                            docs.append({"title": title, "url": url, "snippet": snippet})
            except Exception as e:
                # 联网搜索失败不应中断整个知识库查询
                logger.warning(f"联网搜索失败，将跳过网络结果: {e}")

        if docs:
            logger.info(f"MCP 搜索结果数量: {len(docs)}")
        else:
            logger.info("联网搜索无结果，将仅使用本地知识库检索结果")

        # 始终显式返回列表，保证下游可以安全遍历
        return {"web_search_docs": docs}

    async def _mcp_call(self, query):

        search_mcp = MCPServerStreamableHttp(
            name="search_mcp",
            params={
                "url": mcp_config.mcp_base_url,
                "headers": {"Authorization": f"Bearer {mcp_config.api_key}"},
                "timeout": 10,
            },
            cache_tools_list=True,
            max_retry_attempts=3,
        )

        try:
            await search_mcp.connect()
            result = await search_mcp.call_tool(
                tool_name="bailian_web_search",
                arguments={"query": query, "count": 5},
            )
            return result
        finally:
            await search_mcp.cleanup()












if __name__ == "__main__":

    init_state = {
        "rewritten_query": "关于brother HAK180烫金机，如何调节转印温度？"
    }

    # 执行节点的业务调用
    node_web_search_mcp = NodeWebSearchMcp()
    result = node_web_search_mcp(init_state)
    logger.info(serialize_json(result, indent=4))




