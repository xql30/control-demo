"""Bilingual conversational movie demo."""
import json
import os
from pathlib import Path
from dotenv import load_dotenv
ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / '.env')
for name, value in [('DEVICE','cpu'),('OMP_NUM_THREADS','2'),('MKL_NUM_THREADS','2'),('TOKENIZERS_PARALLELISM','false')]:
    os.environ.setdefault(name, value)
import streamlit as st
import pandas as pd
st.set_page_config(page_title='ControlRec Demo', page_icon='🎬', layout='wide',
                   menu_items={'Get Help':None,'Report a bug':None,'About':None})
try:
    for name in ['LLM_BASE_URL','LLM_MODEL','LLM_API_KEY','HF_TOKEN','ASSET_REPO_ID','ASSET_REVISION']:
        if name in st.secrets: os.environ[name] = str(st.secrets[name])
except (FileNotFoundError, st.errors.StreamlitSecretNotFoundError):
    pass
with st.sidebar:
    language = st.selectbox('语言 / Language', ['中文','English'], key='language')
lang = 'zh' if language == '中文' else 'en'
def t(zh, en): return zh if lang == 'zh' else en
@st.cache_resource(show_spinner=False)
def load_engine():
    import torch
    torch.set_num_threads(2)
    from demo.assets import ensure_assets
    from demo.movie_backend import MovieEngine
    return MovieEngine(ensure_assets())
def reset_chat():
    st.session_state.update(messages=[], result=None, preference_cache={}, previous=[])
if 'messages' not in st.session_state: reset_chat()
st.title(t('ControlRec · 电影对话','ControlRec · Movie conversations'))
st.caption(t('用自己的话描述此刻想看的电影，随时追加、排除或修改偏好。',
             'Describe what you feel like watching. Add, exclude, or change preferences as you chat.'))
try:
    with st.spinner(t('正在加载推荐模型…','Loading the recommender…')): engine = load_engine()
except Exception:
    st.error(t('推荐服务暂时未能加载，请稍后重试。','The recommender could not load. Please try again later.'))
    st.stop()
showcase = json.loads((ROOT/'examples/showcase.json').read_text())
def try_case(key):
    reset_chat()
    st.session_state.history = [m['id'] for m in showcase['history']]
    st.session_state.active_history = list(st.session_state.history)
    case = next(c for c in showcase['cases'] if c['key']==key)
    if key == 'history': st.session_state.pending_history = True
    else: st.session_state.prompt_input = case['prompt'][lang]
def queue_history():
    reset_chat()
    st.session_state.pending_history = True
with st.sidebar:
    st.subheader(t('观影历史','Viewing history'))
    selected = st.multiselect(t('已看过的电影（按观看顺序选择）','Films you have seen (in viewing order)'),
        options=list(engine.item2id), format_func=lambda i:engine.movie(i)['title'], max_selections=20, key='history')
    if selected != st.session_state.get('active_history',[]):
        reset_chat(); st.session_state.active_history = list(selected)
    st.button(t('仅根据历史推荐','Recommend from history'), on_click=queue_history, disabled=not selected, width='stretch')
    st.button(t('重置对话','Reset conversation'), on_click=reset_chat, width='stretch')
    with st.expander(t('高级设置','Advanced settings')):
        strength = st.slider(t('偏好引导强度','Preference guidance strength'),0.,8.,3.,.5,key='strength')
    st.caption(t('对话及所选观影历史会由在线服务处理，用于生成推荐。',
                 'Your conversation and selected viewing history are processed by online services to generate recommendations.'))
with st.expander(t('同一观影历史，不同当前需求 · 看看示例','Same history, different needs · Explore examples'), expanded=True):
    st.caption(t('已看过：','Previously watched: ')+' · '.join(m['title'] for m in showcase['history']))
    for col, case in zip(st.columns(3), showcase['cases']):
        with col:
            st.markdown('**'+case['prompt'][lang]+'**')
            featured = case.get('featured_movie')
            if case['key']=='history':
                st.success(t('历史推荐首位：','Top history-based result: ')+case['movies'][0]['title'])
                st.write(t('未添加当前需求；下方是仅依据观影历史生成的结果。','No current request was added. These results were generated from viewing history alone.'))
            else:
                if featured: st.success(t('本轮重点推荐：','Featured recommendation: ')+featured['title'])
                with st.chat_message('assistant'): st.write(case['assistant_reply_i18n'][lang])
            with st.expander(t('完整推荐列表','Full recommendation list')):
                for n,m in enumerate(case['movies'],1): st.write(f"{n}. {m['title']}")
            st.button(t('试试这个需求','Try this request') if case['key']!='history' else t('试试仅按历史推荐','Try history-only recommendations'),
                      key='example_'+case['key'], on_click=try_case, args=(case['key'],), width='stretch')
    st.caption(t('以上为同一历史下三次独立运行的实际记录；后两列包含当时返回的推荐回复。点击示例可继续体验，实时回复可能不同。',
                 'Recorded outputs from three independent runs with the same history. The last two include the returned explanations. Try a request to continue; live responses may differ.'))
if st.session_state.pop('pending_history',False):
    try:
        with st.spinner(t('正在根据观影历史生成推荐…','Generating recommendations from history…')):
            st.session_state.result = engine.history_only(selected)
    except Exception:
        st.error(t('推荐暂时不可用，请稍后重试。','Recommendations are temporarily unavailable. Please try again.'))
left,right = st.columns([.9,1.1],gap='large')
with left:
    st.subheader(t('和推荐助手聊聊','Chat with your recommender'))
    with st.container(height=420):
        if not st.session_state.messages:
            st.info(t('选择一个示例，或直接说说当前需求。之后可以继续追加条件。','Choose an example or describe what you want to watch, then refine your request.'))
        for message in st.session_state.messages:
            with st.chat_message(message['role']): st.write(message.get('i18n',{}).get(lang,message['content']))
    prompt = st.chat_input(t('说说你现在想看什么…','What would you like to watch?'),max_chars=2000,key='prompt_input')
    if prompt:
        messages = st.session_state.messages+[{'role':'user','content':prompt}]
        if len(messages)>24:
            st.error(t('本次对话已达到 12 轮，请重置后继续。','This conversation has reached 12 turns. Reset to continue.'))
        else:
            try:
                with st.spinner(t('正在理解需求并生成推荐…','Understanding your request and generating recommendations…')):
                    result = engine.chat(selected,[{'role':m['role'],'content':m['content']} for m in messages],strength,
                        preference_cache=st.session_state.preference_cache,language=lang)
                st.session_state.previous = [m['id'] for m in (st.session_state.result or {}).get('movies',[])]
                st.session_state.messages = messages+[{'role':'assistant','content':result['assistant_reply'],'i18n':result['assistant_reply_i18n']}]
                st.session_state.result = result
                st.rerun()
            except Exception:
                st.error(t('本轮推荐未完成，请稍后重试或调整需求。','This request could not be completed. Please retry or adjust your request.'))
with right:
    st.subheader(t('为你生成的推荐','Your recommendations'))
    result = st.session_state.result
    if result:
        tabs = st.tabs([t('推荐列表','Recommendations'),t('当前偏好','Current preferences')])
        with tabs[0]:
            featured = result.get('featured_movie')
            if featured: st.success(t('本轮重点推荐：','Featured recommendation: ')+featured['title'])
            if result.get('history_only'): st.caption(t('仅根据观影历史生成，未添加当前需求。','Generated from viewing history with no current request.'))
            rows = []
            for n,m in enumerate(result['movies'],1):
                row={t('排名','Rank'):n,t('电影','Movie'):m['title']}
                if st.session_state.previous: row[t('变化','Change')] = t('保留','Retained') if m['id'] in st.session_state.previous else t('新增','New')
                rows.append(row)
            st.dataframe(pd.DataFrame(rows),hide_index=True,width='stretch')
            st.caption(t(f"共 {len(rows)} 部电影",f"{len(rows)} movies"))
        with tabs[1]:
            if result.get('history_only'): st.write(t('尚未添加当前需求。','No current request has been added.'))
            else:
                labels=result.get('display_labels',{}); state=result['state']
                positives = labels.get('positive_labels_zh') if lang=='zh' else None
                negatives = labels.get('negative_labels_zh') if lang=='zh' else None
                st.markdown('**'+t('想看','Looking for')+'**')
                for item in positives or [i['text'] for i in state['intents']]: st.write('• '+item)
                st.markdown('**'+t('暂时不想看','Avoiding')+'**')
                for item in negatives or state['negative_intents']: st.write('• '+item)
                if not state['negative_intents']: st.write(t('暂无排除要求','No exclusions'))
    else:
        st.info(t('发送需求，或点击“仅根据历史推荐”，即可查看结果。','Send a request or choose “Recommend from history” to see results.'))
