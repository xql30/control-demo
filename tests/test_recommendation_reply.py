import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from demo.recommendation_reply import explain_recommendation


class RecommendationReplyTest(unittest.TestCase):
    def run_reply(self, level='partial', selected_id='1', violation=False):
        movie = {'id': '1', 'title': 'Example film', 'genres': ['Comedy'],
                 'violation': violation}
        reply = {'selected_id': selected_id, 'match_level': level,
                 'reason_zh': '有轻松的日常片段。但也有一些冲突。',
                 'reason_en': 'It has light everyday moments. Some conflict remains.',
                 'positive_labels_zh': ['轻松'], 'negative_labels_zh': []}
        engine = SimpleNamespace(catalog={'1': {'plot': 'Light everyday moments and conflict.'}},
                                 llm=SimpleNamespace(complete=Mock(return_value=json.dumps(reply))))
        result = explain_recommendation(engine, [], [{'role': 'user', 'content': '轻松一点'}],
            {'movies': [movie], 'state': {'intents': [{'text': 'Lighthearted'}],
                                        'negative_intents': []}})
        return result, engine.llm.complete

    def test_partial_match_is_selected_and_labeled_in_both_languages(self):
        result, complete = self.run_reply()
        self.assertEqual(result['featured_movie']['id'], '1')
        self.assertEqual(result['match_level'], 'partial')
        self.assertIn('部分符合', result['assistant_reply_i18n']['zh'])
        self.assertIn('partial match', result['assistant_reply_i18n']['en'])
        self.assertEqual(complete.call_count, 1)

    def test_full_match_remains_supported(self):
        result, _ = self.run_reply(level='full')
        self.assertEqual(result['match_level'], 'full')
        self.assertNotIn('部分符合', result['assistant_reply'])

    def test_no_match_does_not_force_a_selection(self):
        result, _ = self.run_reply(level='none', selected_id=None)
        self.assertIsNone(result['featured_movie'])
        self.assertEqual(result['match_level'], 'none')

    def test_known_violation_cannot_be_selected_as_partial(self):
        result, complete = self.run_reply(violation=True)
        self.assertIsNone(result['featured_movie'])
        self.assertEqual(result['reply_status'], 'fallback')
        self.assertEqual(complete.call_count, 2)

    def test_invented_id_is_rejected(self):
        result, _ = self.run_reply(selected_id='999')
        self.assertIsNone(result['featured_movie'])
        self.assertEqual(result['reply_status'], 'fallback')

    def test_inconsistent_or_invalid_match_level_is_rejected(self):
        for level, selected_id in [('none', '1'), ('partial', None), ('anything', '1')]:
            with self.subTest(level=level, selected_id=selected_id):
                result, _ = self.run_reply(level=level, selected_id=selected_id)
                self.assertIsNone(result['featured_movie'])
                self.assertEqual(result['match_level'], 'unknown')


if __name__ == '__main__':
    unittest.main()
