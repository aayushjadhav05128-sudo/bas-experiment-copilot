"""
BAS SSV2 Action Recognition Pipeline
Bharatiya Antariksh Station (BAS) Payload Operations Copilot
"""

CLASSES = [
    "open_payload",
    "retrieve_object",
    "inspect_proxy",
    "return_object",
    "close_payload"
]

LABEL_TO_IDX = {cls_name: idx for idx, cls_name in enumerate(CLASSES)}
IDX_TO_LABEL = {idx: cls_name for idx, cls_name in enumerate(CLASSES)}

# SSV2 to BAS Label Mapping
SSV2_LABEL_MAP = {
    "Opening something": "open_payload",
    "Uncovering something": "open_payload",
    "Taking something out of something": "retrieve_object",
    "Taking something from somewhere": "retrieve_object",
    "Holding something": "inspect_proxy",
    "Putting something into something": "return_object",
    "Closing something": "close_payload",
}
