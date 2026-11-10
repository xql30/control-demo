import copy
import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from demo.conversation import handle_turn


class ConversationTest(unittest.TestCase):
    def setUp(self):
        self.current = {
            'movies': [{'id': '1', 'title': 'Example film', 'genres': ['Comedy']}],
            'state': {'intents': [{'text': 'Lighthearted movies'}]},
            'preference_messages': [{'role': 'user', 'content': '推荐轻松的电影'}]}
        self.engine = SimpleNamespace(catalog={'1': {'plot': 'Two friends go on a trip.'}},
                                      llm=SimpleNamespace(complete=Mock()), chat=Mock())
        self.engine.chat.return_value = {'movies': [], 'assistant_reply_i18n':
                                        {'zh': '更新的推荐', 'en': 'Updated recommendations'}}

    def decision(self, action, zh='', en=''):
        self.engine.llm.complete.return_value = json.dumps(
            {'action': action, 'answer_zh': zh, 'answer_en': en})

    def test_question_answers_without_regeneration_or_state_mutation(self):
        self.decision('answer', '讲述两个朋友的旅行。', 'It follows two friends on a trip.')
        before = copy.deepcopy(self.current)
        cache = {'existing': 'unchanged'}
        result = handle_turn(self.engine, [], [{'role': 'user', 'content': '第一部讲什么？'}],
                             self.current, preference_cache=cache)
        self.engine.chat.assert_not_called()
        self.assertEqual(result['response_type'], 'answer')
        self.assertEqual(self.current, before)
        self.assertEqual(cache, {'existing': 'unchanged'})
        payload = json.loads(self.engine.llm.complete.call_args.args[0][1]['content'])
        self.assertEqual(payload['current_candidates'][0]['plot'], 'Two friends go on a trip.')

    def test_later_preference_edit_does_not_parse_intervening_question(self):
        self.decision('recommend')
        messages = [{'role': 'user', 'content': '第一部讲什么？'},
                    {'role': 'assistant', 'content': '两个朋友的旅行。'},
                    {'role': 'user', 'content': '换一批，不要恐怖片'}]
        result = handle_turn(self.engine, [], messages, self.current)
        expected = self.current['preference_messages'] + [messages[-1]]
        self.assertEqual(self.engine.chat.call_args.args[1], expected)
        self.assertEqual(result['preference_messages'], expected)
        self.assertEqual(len(self.current['preference_messages']), 1)

    def test_question_without_previous_recommendation(self):
        self.decision('answer', '可以描述你现在想看的内容。', 'Describe what you feel like watching.')
        result = handle_turn(self.engine, [], [{'role': 'user', 'content': '怎么用？'}], language='en')
        self.engine.chat.assert_not_called()
        self.assertEqual(result['assistant_reply'], 'Describe what you feel like watching.')

    def test_mixed_question_and_request_preserves_both_parts(self):
        self.decision('recommend', '这部电影讲旅行。', 'This film is about a trip.')
        result = handle_turn(self.engine, [], [{'role': 'user', 'content': '这部讲什么？再推荐几部类似的'}], self.current)
        self.assertEqual(result['response_type'], 'recommendation')
        self.assertEqual(result['assistant_reply'], '这部电影讲旅行。\n\n更新的推荐')

    def test_invalid_route_never_defaults_to_recommending(self):
        self.decision('invalid')
        with self.assertRaises(ValueError):
            handle_turn(self.engine, [], [{'role': 'user', 'content': '谢谢'}], self.current)
        self.engine.chat.assert_not_called()


if __name__ == '__main__':
    unittest.main()
