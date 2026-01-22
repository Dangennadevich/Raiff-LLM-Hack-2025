import logging

def get_node_logger(node_name: str) -> logging.Logger:
    return logging.getLogger(f"agent.node.{node_name}")