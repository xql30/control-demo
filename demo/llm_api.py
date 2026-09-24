"""OpenAI-compatible API with per-session credentials and safe custom endpoints."""
import ipaddress
import os
import socket
from urllib.parse import urlsplit
import requests

class LLMError(RuntimeError):
    pass


def validate_custom_endpoint(url):
    parts=urlsplit(url)
    if parts.scheme != 'https' or not parts.hostname or parts.username or parts.password or parts.query or parts.fragment:
        raise LLMError('自定义接口须为不含账号、查询参数的 HTTPS Base URL。')
    if parts.port not in (None,443):
        raise LLMError('自定义接口仅支持 HTTPS 标准端口 443。')
    try:
        addresses=socket.getaddrinfo(parts.hostname,443,type=socket.SOCK_STREAM)
    except OSError:
        raise LLMError('无法解析自定义接口域名。') from None
    if not addresses or any(not ipaddress.ip_address(row[4][0]).is_global for row in addresses):
        raise LLMError('自定义接口必须使用公网地址，不支持内网或本机地址。')
    return url.rstrip('/')


class ChatAPI:
    def __init__(self,base_url=None,model=None,key=None,custom=False):
        self.base_url=(base_url if base_url is not None else os.getenv('LLM_BASE_URL','https://coding.dashscope.aliyuncs.com/v1')).rstrip('/')
        self.model=model if model is not None else os.getenv('LLM_MODEL','qwen3.5-plus')
        self.key=key if key is not None else os.getenv('LLM_API_KEY','')
        self.custom=custom
        self.timeout=float(os.getenv('LLM_TIMEOUT','60'))
        if custom:
            if not key: raise LLMError('使用自定义接口时，请填写自己的 API Key。')
            validate_custom_endpoint(self.base_url)

    def complete(self,messages,max_tokens=600):
        if not self.key: raise LLMError('默认 API 尚未配置，可在侧栏使用自己的 API。')
        if self.custom: validate_custom_endpoint(self.base_url)
        payload={'model':self.model,'messages':messages,'temperature':0,'max_tokens':max_tokens,'stream':False}
        hostname=urlsplit(self.base_url).hostname or ''
        if hostname.endswith('.dashscope.aliyuncs.com') or hostname=='dashscope.aliyuncs.com':
            payload['enable_thinking']=False
        try:
            response=requests.post(self.base_url+'/chat/completions',headers={'Authorization':'Bearer '+self.key},
                json=payload,timeout=(10,self.timeout),allow_redirects=False)
        except requests.RequestException:
            raise LLMError('语言服务暂时无法连接，请稍后重试。') from None
        if response.status_code!=200:
            raise LLMError(f'语言服务请求失败（HTTP {response.status_code}），请检查接口、模型或额度。')
        try:
            content=response.json()['choices'][0]['message']['content']
            if not isinstance(content,str) or not content.strip(): raise ValueError()
            return content
        except (ValueError,KeyError,IndexError,TypeError):
            raise LLMError('语言服务返回格式异常，请稍后重试。') from None
