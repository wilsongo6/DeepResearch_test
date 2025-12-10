import json
from concurrent.futures import ThreadPoolExecutor
from typing import List, Union
import requests
from qwen_agent.tools.base import BaseTool, register_tool
import asyncio
from typing import Dict, List, Optional, Union
import uuid
import http.client
import json
import time
import os

@register_tool("search", allow_overwrite=True)
class Search(BaseTool):
    name = "search"
    description = "Performs batched web searches: supply an array 'query'; the tool retrieves the top 10 results for each query in one call."
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "array",
                "items": {
                    "type": "string"
                },
                "description": "Array of query strings. Include multiple complementary search queries in a single call."
            },
        },
        "required": ["query"],
    }

    def __init__(self, cfg: Optional[dict] = None):
        super().__init__(cfg)

    def google_search_with_serp(self, query: str):
        base_url = "https://reader.psmoe.com/glm/s/"
        max_retries = 5
        for attempt in range(max_retries):
            try:
                import urllib.parse
                query_encoded = urllib.parse.quote(query)
                search_url = f"{base_url}?q={query_encoded}&provider=serper"
                response = requests.get(search_url, timeout=30, verify=False)
                response.raise_for_status()
                return response.text
            except Exception as e:
                if attempt < 4:  # 不是最后一次，等待后重试
                    time.sleep(2 ** attempt)  # 指数退避: 1s, 2s, 4s, 8s
                    continue
                else:  # 最后一次失败
                    return f"[Search] Failed after 5 attempts: {str(e)}"

    def search_with_serp(self, query: str):
        result = self.google_search_with_serp(query)
        return result


    def call(self, params: Union[str, dict], **kwargs) -> str:
        # 参数校验
        try:
            query = params["query"]
        except:
            return "[Search] Invalid request format: Input must be a JSON object containing 'query' field"

        # 单个查询
        if isinstance(query, str):
            response = self.search_with_serp(query)

        # 多个查询
        else:
            assert isinstance(query, List)
            responses = []
            for q in query:
                responses.append(self.search_with_serp(q))
            response = "\n=======\n".join(responses)

        return response
