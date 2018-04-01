import unittest

from dengue_prediction.models.metrics import Metric, MetricList

class TestMetric(unittest.TestCase):

    def setUp(self):
        name    = "Accuracy"
        scoring = "accuracy"
        value   = 0.0
        self.metric = Metric(name, scoring, value)

    def test_conversion_user(self):
        self.assertEqual(
            self.metric,
            Metric.from_dict(self.metric.convert(kind="user"), kind="user"))

    def test_conversion_db(self):
        self.assertEqual(
            self.metric,
            Metric.from_dict(self.metric.convert(kind="db"), kind="db"))

class TestMetricList(unittest.TestCase):

    def setUp(self):
        self.metric_list1 = MetricList()
        self.metric_list2 = MetricList()

    def test_empty_list_semantics(self):
        # create list
        self.assertEqual(self.metric_list1, self.metric_list2)
        self.assertEqual(self.metric_list1, self.metric_list2)
        self.assertIsNot(self.metric_list1, self.metric_list2)

    def _add_elements(self, list):
        list.append(Metric("accuracy", "accuracy", 0.0))
        list.append(Metric("precision", "precision", 0.5))

    def test_add_elements(self):
        # add elements
        self._add_elements(self.metric_list1)
        self._add_elements(self.metric_list2)

        # check equality operations
        self.assertEqual(self.metric_list1, self.metric_list1)
        self.assertEqual(self.metric_list1, self.metric_list2)

        # check non-equality
        self.metric_list1.append(Metric("Recall", "recall", 0.7))
        self.assertNotEqual(self.metric_list1, self.metric_list2)

        self.metric_list2[1] = Metric("Precision", "precision", 0.4)
        self.assertNotEqual(self.metric_list1, self.metric_list2)

    @unittest.expectedFailure
    def test_inverses(self):
        self._add_elements(self.metric_list1)
        self.metric_list1.append(Metric("Recall", "recall", 0.7))

        for kind, method in [('user', 'from_dict_user'),
                             ('db', 'from_list_db')]:
            converted = self.metric_list1.convert(kind="user")
            inverted = getattr(MetricList, method)(converted)
            self.assertEqual(self.metric_list1, inverted)
            inverted = MetricList.from_object(converted)
            self.assertEqual(self.metric_list1, inverted)
