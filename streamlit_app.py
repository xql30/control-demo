"""Anonymous-ready ControlRec demo with per-session API configuration."""
import os
from pathlib import Path
from dotenv import load_dotenv

ROOT=Path(__file__).resolve().parent
load_dotenv(ROOT/'.env')
os.environ.setdefault('DEVICE','cpu')
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')

import streamlit as st
import pandas as pd
from demo.llm_api import ChatAPI,LLMError

st.set_page_config(page_title='ControlRec Demo',page_icon='🎬',layout='wide',menu_items={'Get Help':None,'Report a bug':None,'About':None})
try:
    for name in ['LLM_BASE_URL','LLM_MODEL','LLM_API_KEY','HF_TOKEN','ASSET_REPO_ID','ASSET_REVISION']:
        if name in st.secrets: os.environ[name]=str(st.secrets[name])
except (FileNotFoundError,st.errors.StreamlitSecretNotFoundError):
    pass

@st.cache_resource(show_spinner='正在加载推荐模型…')
def load_engine():
    import torch
    torch.set_num_threads(2)
    from demo.assets import ensure_assets
    from demo.movie_backend import MovieEngine
    return MovieEngine(ensure_assets())


def reset_chat():
    st.session_state.messages=[]
    st.session_state.result=None
    st.session_state.preference_cache={}
    st.session_state.previous=[]

if 'messages' not in st.session_state: reset_chat()

st.title('ControlRec · 电影对话')
st.caption('用自己的话描述想看的电影，随时追加、排除或修改偏好。')
try:
    engine=load_engine()
except Exception:
    st.error('模型暂时未能加载，请由部署者检查模型资产与访问权限。')
    st.stop()

with st.sidebar:
    st.subheader('对话设置')
    selected=st.multiselect('已看过的电影（按观看顺序选择）',options=list(engine.item2id),
        format_func=lambda i:engine.movie(i)['title'],max_selections=20,key='history')
    if selected != st.session_state.get('active_history',[]):
        reset_chat()
        st.session_state.active_history=list(selected)
    if st.button('重置对话',width='stretch'): reset_chat()
    with st.expander('语言模型 API',expanded=False):
        mode=st.radio('接口来源',['默认接口','使用自己的接口'],key='api_mode')
        if mode=='使用自己的接口':
            base_url=st.text_input('Base URL',value='https://dashscope.aliyuncs.com/compatible-mode/v1',key='custom_url')
            model=st.text_input('模型名称',value='qwen-plus',key='custom_model')
            key=st.text_input('API Key',type='password',key='custom_key')
            st.caption('密钥仅用于当前会话，不写入代码或日志。')
        else:
            base_url=os.getenv('LLM_BASE_URL','https://coding.dashscope.aliyuncs.com/v1')
            model=os.getenv('LLM_MODEL','qwen3.5-plus')
            key=None
            st.caption('当前模型：'+model)
    with st.expander('高级设置'):
        strength=st.slider('偏好引导强度',0.,8.,3.,.5)
    st.caption('请求、所选观影历史及候选电影信息会发送至当前语言模型服务。推荐列表由推荐模型生成。')

signature=(mode,base_url,model)
if signature!=st.session_state.get('api_signature'):
    st.session_state.preference_cache={}
    st.session_state.api_signature=signature

left,right=st.columns([.9,1.1],gap='large')
with left:
    st.subheader('和推荐助手聊聊')
    conversation=st.container(height=500)
    with conversation:
        if not st.session_state.messages:
            st.info('例如：想看治愈、轻松一点的电影，不要超英题材。之后可以追加“再来一些喜剧”，或撤销之前的排除。')
        for message in st.session_state.messages:
            with st.chat_message(message['role']): st.write(message['content'])
    prompt=st.chat_input('说说你现在想看什么…',max_chars=2000)
    if prompt:
        messages=st.session_state.messages+[{'role':'user','content':prompt}]
        try:
            client=ChatAPI(base_url=base_url,model=model,key=key,custom=mode=='使用自己的接口')
            with st.spinner('正在理解需求并生成推荐…'):
                result=engine.chat(selected,messages,strength,llm=client,
                    preference_cache=st.session_state.preference_cache)
            st.session_state.previous=[m['id'] for m in (st.session_state.result or {}).get('movies',[])]
            st.session_state.messages=messages+[{'role':'assistant','content':result['assistant_reply']}]
            st.session_state.result=result
            st.rerun()
        except (LLMError,ValueError) as error:
            st.error(str(error))
        except Exception:
            st.error('推荐服务暂时不可用，请稍后重试。')

with right:
    st.subheader('为你生成的推荐')
    result=st.session_state.result
    if result:
        tabs=st.tabs(['推荐列表','当前偏好','意图分组'])
        with tabs[0]:
            featured=result.get('featured_movie')
            if featured: st.success('本轮重点推荐：'+featured['title'])
            rows=[]
            for n,movie in enumerate(result['movies'],1):
                status='保留' if movie['id'] in st.session_state.previous else '新增'
                rows.append({'排名':n,'电影':movie['title'],'类型':' / '.join(movie['genres']),
                    '变化':status,'排除条件':'类型违规' if movie['violation'] else ('自由语义待核验' if result['state']['negative_intents'] else '未命中排除类型')})
            st.dataframe(pd.DataFrame(rows),hide_index=True,width='stretch')
            st.caption(f"{len(rows)} 部电影 · {len(result['branches'])} 个意图 · 本轮 {result['total_seconds']:.1f} 秒")
        with tabs[1]:
            state=result['state']
            st.markdown('**想看**')
            for intent in state['intents']: st.write('• '+intent['text'])
            st.markdown('**暂时不想看**')
            st.write('、'.join(state['negative_intents']) or '暂无排除要求')
            if state.get('unresolved'): st.info('；'.join(state['unresolved']))
        with tabs[2]:
            for branch in result['branches']:
                with st.expander(branch['intent']['text']):
                    for movie in branch['movies']: st.write(movie['title'])
    else:
        st.info('发送第一条需求后，这里会显示 10 部推荐电影。')
