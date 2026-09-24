"""A grounded conversational explanation of one of the generated movies."""
import json
import re
import time
import torch


def explain_recommendation(engine, history, messages, result):
    candidates = []
    for movie in result['movies']:
        info = engine.catalog.get(movie['id'], {})
        candidates.append({'id': movie['id'], 'title': movie['title'], 'genres': movie['genres'],
                           'plot': info.get('plot', '')[:1400], 'director': info.get('director', []),
                           'cast': info.get('star', []), 'known_genre_violation': movie['violation']})
    state = result['state']
    payload = {'history': [engine.movie(i) for i in history],
               'latest_request': [m['content'] for m in messages if m['role'] == 'user'][-1],
               'current_positive': state['intents'], 'current_negative': state['negative_intents'],
               'excluded_genres': state['excluded_genres'], 'candidates': candidates}
    prompt = (
        '你是温和、简洁的电影小助手。推荐系统已经生成了候选列表。请从中挑一部相对最符合当前需求的电影作重点介绍。'
        '只能选candidates中的id，不得另造电影，也不要修改推荐列表。优先避开明确不符合排除要求的电影。'
        '根据简介选择电影，但回复不要复述具体情节、人物、地点或时间；只简短解释其已知类型为何相对贴近用户这次需求。不要捏造中文译名、观影历史或声称看过电影。'
        '如果这些候选都不理想，selected_id返回null，如实说明，不必强推。'
        '只输出JSON对象，字段selected_id和reason。reason为两句自然中文，约40到100字，只解释理由或不足，'
        '不要写电影名（片名由程序准确插入），不要列编号，不要写“已记录本轮偏好修改”。'
        '不要保证绝对最满足或所有推荐都满足。没有观影历史时不要声称结合了历史。'
        '不要套用固定免责声明；只有真实冲突才提示不足。用户没说排除剧情时，不要把剧情元素说成缺点。语气像在帮朋友挑一部片，而不是写分析报告。')
    conversation = [{'role': 'system', 'content': prompt},
                    {'role': 'user', 'content': json.dumps(payload, ensure_ascii=False)}]
    start = time.perf_counter()
    selected = None
    reason = None
    status = 'api'
    for attempt in range(2):
        raw = engine.llm.complete(conversation, max_tokens=500)
        try:
            reply = json.loads(raw[raw.index('{'):raw.rindex('}')+1])
            ident = reply.get('selected_id')
            reason = reply['reason']
            if not isinstance(reason, str) or not reason.strip() or len(reason) > 600:
                raise ValueError('reason必须是简短中文说明')
            if ident is not None:
                selected = next((m for m in result['movies'] if m['id'] == str(ident)), None)
                if selected is None:
                    raise ValueError('selected_id不在候选列表中')
                if selected['violation']:
                    raise ValueError('不要重点推荐已知违反排除类型的电影；没有合适的请选择null')
            break
        except (ValueError, TypeError, KeyError):
            if attempt == 0:
                conversation += [{'role': 'assistant', 'content': raw}, {'role': 'user', 'content':
                    '请修正格式：selected_id只能是候选id或null；reason必须是中文字符串。不要选择已标记类型违规的电影。'}]
            else:
                selected = None
                status = 'safe_fallback'
                reason = '这次列出了10部候选，你可以先看看片名和类型；重点推荐的说明暂时没生成好，我先不替你下判断。'
    # The catalog supplies the sole displayed title; discard generated aliases.
    reason = re.sub(r'《[^》]*》', '它', reason).replace('用户', '你').replace('该电影', '它')
    intro = ('已结合您的观影历史和当前需求更新了这10部推荐。' if history
             else '已根据您当前的需求更新了这10部推荐。')
    if selected is not None:
        reply_text = intro + '\n\n这一组里，我更推荐《' + selected['title'] + '》。' + reason.strip()
    else:
        reply_text = intro + '\n\n' + reason.strip()
    return {'assistant_reply': reply_text, 'featured_movie': selected,
            'reply_status': status, 'reply_seconds': time.perf_counter()-start}
