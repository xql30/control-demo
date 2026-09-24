# ControlRec Demo

An interactive movie recommender with free-form conversational preference editing.
The recommendation model runs on CPU. An OpenAI-compatible language-model API parses requests and explains generated recommendations. Visitors may supply their own endpoint, model, and API key in the sidebar.

## Streamlit Community Cloud

Connect a private GitHub repository to Streamlit Community Cloud. Set the entrypoint to `streamlit_app.py` and Python to **3.10**. Set the deployed app visibility to public if reviewers should access it without signing in.

In **Advanced settings → Secrets**, configure:

```toml
LLM_BASE_URL = "https://coding.dashscope.aliyuncs.com/v1"
LLM_MODEL = "qwen3.5-plus"
LLM_API_KEY = "<server-side default key>"
HF_TOKEN = "<read-only token for private assets>"
ASSET_REPO_ID = "<private inference asset repository>"
ASSET_REVISION = "<pinned asset commit>"
```

The application uses the configured endpoint; deployment operators should ensure their provider permits the intended use. No API keys, training dialogues, or model assets belong in the GitHub source repository. Inference assets are downloaded server-side from a private model repository.

For double-blind review, use a dedicated anonymous GitHub/Streamlit account, a neutral app URL, and check the deployed app as a logged-out visitor. A private repository alone does not guarantee author anonymity.

## Local run

Install `requirements.txt`, configure `.env`, then run:

```bash
streamlit run streamlit_app.py
```

Open `http://127.0.0.1:8501`. Local execution does not create a public tunnel.
