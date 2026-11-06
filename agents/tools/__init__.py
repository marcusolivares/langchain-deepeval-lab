"""The tools the bots call, kept apart from the agents that use them:

- shop.py          : get_order_status, get_refund_policy (shared by the agents and the chatbot)
- policy_search.py : search_policies, the RAG agent's retriever over the policy knowledge base
- mcp_server.py    : get_shipping_options, served over MCP in its own process
"""
