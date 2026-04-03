import json
from pathlib import Path

from dz_hypergraph.tools.llm import _StreamingTextRecorder, _aggregate_streamed_chat_response


def test_aggregate_streamed_chat_response_parses_sse_chunks():
    chunks = [
        'data: {"choices":[{"index":0,"delta":{"role":"assistant","content":"Hello"},"finish_reason":null}]}\n\n',
        'data: {"choices":[{"index":0,"delta":{"content":" world"},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":2,"total_tokens":12}}\n\n',
        "data: [DONE]\n\n",
    ]

    response = _aggregate_streamed_chat_response(chunks, model="demo-model")

    assert response["model"] == "demo-model"
    assert response["choices"][0]["message"]["role"] == "assistant"
    assert response["choices"][0]["message"]["content"] == "Hello world"
    assert response["choices"][0]["finish_reason"] == "stop"
    assert response["usage"]["total_tokens"] == 12


def test_aggregate_streamed_chat_response_accepts_plain_json_fallback():
    body = {
        "id": "abc",
        "object": "chat.completion",
        "created": 1,
        "model": "demo-model",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": '{"ok": true}'},
                "finish_reason": "stop",
            }
        ],
        "usage": {"total_tokens": 3},
    }

    response = _aggregate_streamed_chat_response([json.dumps(body)], model="ignored-model")

    assert response["id"] == "abc"
    assert response["choices"][0]["message"]["content"] == '{"ok": true}'
    assert response["usage"]["total_tokens"] == 3


def test_streaming_text_recorder_writes_incremental_text(tmp_path: Path):
    path = tmp_path / "bridge_plan_prose.txt"
    recorder = _StreamingTextRecorder(path)

    recorder.feed('data: {"choices":[{"index":0,"delta":{"role":"assistant","content":"Bridge"}}]}\n')
    recorder.feed('\n')
    recorder.feed('data: {"choices":[{"index":0,"delta":{"content":" plan"}}]}\n\n')
    recorder.feed("data: [DONE]\n\n")
    recorder.finalize()

    assert recorder.saw_sse is True
    assert path.read_text(encoding="utf-8") == "Bridge plan"
