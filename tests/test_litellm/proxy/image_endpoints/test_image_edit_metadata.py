import json

import pytest
from starlette.requests import Request

from litellm import Router
from litellm.litellm_core_utils.core_helpers import get_litellm_metadata_from_kwargs
from litellm.proxy.litellm_pre_call_utils import (
    LiteLLMProxyRequestSetup,
    _get_metadata_variable_name,
)
from litellm.proxy.spend_tracking.spend_tracking_utils import _get_spend_logs_metadata
from litellm.types.utils import ImageResponse


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["/images/edits", "/v1/images/edits"])
async def test_image_edit_header_metadata_survives_router_and_spend_logging(path):
    expected = {"business_type": "image_edit", "debug_id": "metadata-regression"}
    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": path,
            "headers": [
                (b"content-type", b"multipart/form-data; boundary=test"),
                (b"x-litellm-spend-logs-metadata", json.dumps(expected).encode()),
            ],
        }
    )
    metadata_field = _get_metadata_variable_name(request)
    data = {metadata_field: {"user_api_key_user_id": "test-user"}}
    LiteLLMProxyRequestSetup.add_litellm_metadata_from_request_headers(
        headers=dict(request.headers),
        data=data,
        _metadata_variable_name=metadata_field,
    )
    spend_metadata = {}

    async def image_edit_provider(**kwargs):
        spend_metadata.update(
            _get_spend_logs_metadata(
                get_litellm_metadata_from_kwargs({"litellm_params": kwargs})
            )
        )
        return ImageResponse(created=0, data=[])

    router = Router(
        model_list=[
            {
                "model_name": "test-image",
                "litellm_params": {
                    "model": "openai/gpt-image-2",
                    "api_key": "test-key",
                },
            }
        ],
        num_retries=0,
    )
    image_edit = router.factory_function(image_edit_provider, call_type="aimage_edit")
    await image_edit(model="test-image", **data)

    assert spend_metadata["spend_logs_metadata"] == expected
    assert spend_metadata["user_api_key_user_id"] == "test-user"


@pytest.mark.parametrize("path", ["/images/generations", "/v1/images/generations"])
def test_image_generation_keeps_legacy_metadata_field(path):
    request = Request({"type": "http", "method": "POST", "path": path, "headers": []})
    assert _get_metadata_variable_name(request) == "metadata"
