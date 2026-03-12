from unittest.mock import patch

from core import generator


def test_single_test_case_dict_is_wrapped():
    single = {
        'use_case': 'View Timeline on Mobile',
        'test_case': 'Timeline displays user-generated activities',
        'preconditions': ['CRM record exists', 'User is logged in'],
        'test_data': {'crm_record_id': 123, 'user_id': 456},
        'steps': ['Navigate to CRM record', 'Select Timeline tab', 'Verify user-generated activities are displayed'],
        'priority': 'high',
        'tags': ['timeline', 'mobile'],
        'expected_results': ['Timeline displays list of user-generated activities'],
        'actual_results': []
    }

    with patch('core.generator.llm_client.generate', return_value=single):
        results = generator._generate_single_suite(prompt='x', model='m', max_retries=0)

    assert isinstance(results, list)
    assert len(results) == 1
    assert results[0]['use_case'] == single['use_case']
